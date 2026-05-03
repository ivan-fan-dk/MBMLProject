# %%
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pathlib import Path
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split


# Reproducibility
np.random.seed(11)
torch.manual_seed(11)
pyro.set_rng_seed(11)


# %%
def create_long_format(df):
    """Convert Swissmetro wide rows to long format with availability masking."""
    long_data = []

    obs_id_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(df.index.unique())}

    for idx, row in df.iterrows():
        obs_id_seq = obs_id_map[idx]
        person_id = row["ID"]

        if row["TRAIN_AV"] == 1:
            long_data.append(
                {
                    "obs_id": obs_id_seq,
                    "ID": person_id,
                    "alt_id": 0,
                    "tt": row["TRAIN_TT_SCALED"],
                    "co": row["TRAIN_CO_SCALED"],
                    "choice": 1 if row["CHOICE"] == 1 else 0,
                }
            )

        if row["SM_AV"] == 1:
            long_data.append(
                {
                    "obs_id": obs_id_seq,
                    "ID": person_id,
                    "alt_id": 1,
                    "tt": row["SM_TT_SCALED"],
                    "co": row["SM_CO_SCALED"],
                    "choice": 1 if row["CHOICE"] == 2 else 0,
                }
            )

        if row["CAR_AV"] == 1:
            long_data.append(
                {
                    "obs_id": obs_id_seq,
                    "ID": person_id,
                    "alt_id": 2,
                    "tt": row["CAR_TT_SCALED"],
                    "co": row["CAR_CO_SCALED"],
                    "choice": 1 if row["CHOICE"] == 3 else 0,
                }
            )

    long_df = pd.DataFrame(long_data)

    choices_per_obs = long_df.groupby("obs_id")["choice"].sum()
    assert (choices_per_obs == 1).all(), "Some observations have != 1 chosen alternative"

    return long_df


def prepare_lccm_data(long_df, id_col="ID", include_GA=True, reference_Z_cols=None):
    """Build person-level membership matrix Z and long-format arrays for LCCM."""

    def recode_membership_variables(person_df):
        person_df = person_df.copy()

        income_map = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4}
        person_df["INCOME_REC"] = person_df["INCOME"].map(income_map)

        luggage_map = {0: 0, 1: 1, 3: 2}
        person_df["LUGGAGE_REC"] = person_df["LUGGAGE"].map(luggage_map)

        purpose_map = {
            1: 1,
            5: 1,
            2: 2,
            6: 2,
            3: 3,
            7: 3,
            4: 4,
            8: 4,
        }
        person_df["PURPOSE_REC"] = person_df["PURPOSE"].map(purpose_map)

        return person_df

    def build_membership_matrix(person_df, include_GA=False, reference_cols=None):
        person_df = recode_membership_variables(person_df)

        binary_vars = ["MALE", "FIRST"]
        if include_GA:
            binary_vars.append("GA")

        Z = person_df[binary_vars].astype(float).copy()

        for var in ["AGE", "INCOME_REC", "LUGGAGE_REC", "PURPOSE_REC"]:
            dummies = pd.get_dummies(person_df[var], prefix=var, dtype=float)
            dummies = dummies.iloc[:, 1:]
            Z = pd.concat([Z, dummies], axis=1)

        if reference_cols is not None:
            Z = Z.reindex(columns=reference_cols, fill_value=0.0)

        return Z

    df = long_df.copy()
    df["obs_id"] = pd.factorize(df["obs_id"])[0]
    df["alt_id"] = df["alt_id"].astype(int)

    assert (df.groupby("obs_id")["choice"].sum() == 1).all()

    obs_person = (
        df.groupby("obs_id")[id_col]
        .first()
        .reset_index()
        .sort_values("obs_id")
    )
    obs_person["person_idx"] = pd.factorize(obs_person[id_col])[0]

    person_obs_id = obs_person["person_idx"].values
    n_persons = obs_person["person_idx"].nunique()

    demo_cols = [id_col, "AGE", "MALE", "INCOME", "FIRST", "LUGGAGE", "PURPOSE"]
    if include_GA:
        demo_cols.append("GA")

    person_df = df.groupby(id_col, as_index=False).first()[demo_cols].copy()
    person_df["person_idx"] = person_df[id_col].map(
        dict(zip(obs_person[id_col], obs_person["person_idx"]))
    )
    person_df = person_df.sort_values("person_idx").reset_index(drop=True)

    Z_df = build_membership_matrix(
        person_df,
        include_GA=include_GA,
        reference_cols=reference_Z_cols,
    )

    return {
        "df": df,
        "n_obs": df["obs_id"].nunique(),
        "n_alts": df["alt_id"].nunique(),
        "n_persons": n_persons,
        "tt": df["tt"].values,
        "co": df["co"].values,
        "obs_id": df["obs_id"].values,
        "alt_id": df["alt_id"].values,
        "choice": df["choice"].values,
        "person_obs_id": person_obs_id,
        "person_ids": person_df[id_col].values,
        "Z": Z_df.values.astype(float),
        "Z_cols": Z_df.columns.tolist(),
    }


def to_torch_lccm(data):
    return {
        "X": torch.tensor(np.column_stack([data["tt"], data["co"]]), dtype=torch.float32),
        "obs_id": torch.tensor(data["obs_id"], dtype=torch.long),
        "alt_id": torch.tensor(data["alt_id"], dtype=torch.long),
        "choice": torch.tensor(data["choice"], dtype=torch.float32),
        "person_obs_id": torch.tensor(data["person_obs_id"], dtype=torch.long),
        "Z": torch.tensor(data["Z"], dtype=torch.float32),
        "n_obs": data["n_obs"],
        "n_alts": data["n_alts"],
        "n_persons": data["n_persons"],
        "person_ids": data["person_ids"],
        "Z_cols": data["Z_cols"],
    }


# %%
def lccm_pyro_model(
    X,
    obs_id,
    alt_id,
    choice,
    person_obs_id,
    Z,
    n_obs,
    n_alts,
    n_persons,
    K,
    prior_withconstraints,
):
    """Bayesian LCCM with person-level membership model and class-specific utilities."""
    n_features = X.shape[1]

    def matern32_kernel(X1, X2, kappa, lengthscale):
        dist_matrix = torch.cdist(X1, X2)
        scaled = np.sqrt(3.0) * dist_matrix / lengthscale
        return (kappa**2) * (1.0 + scaled) * torch.exp(-scaled)

    kappa = pyro.sample(
        "kappa",
        dist.LogNormal(0.0, 0.5).expand([K]).to_event(1),
    )
    lengthscale = pyro.sample(
        "lengthscale",
        dist.LogNormal(0.0, 0.5).expand([K]).to_event(1),
    )

    f_list = []
    eye = torch.eye(n_persons, dtype=Z.dtype, device=Z.device)
    for k in range(K):
        cov = matern32_kernel(Z, Z, kappa[k], lengthscale[k]) + 1e-4 * eye
        f_list.append(
            pyro.sample(
                f"gp_latent_{k}",
                dist.MultivariateNormal(
                    torch.zeros(n_persons, dtype=Z.dtype, device=Z.device),
                    covariance_matrix=cov,
                ),
            )
        )

    f = torch.stack(f_list, dim=1)
    log_class_probs = torch.log_softmax(f, dim=1)

    if prior_withconstraints:
        beta_raw = pyro.sample(
            "beta_raw",
            dist.Normal(0.0, 1.0).expand([K, n_features]).to_event(2),
        )
        beta = -torch.nn.functional.softplus(beta_raw)
    else:
        beta = pyro.sample(
            "beta",
            dist.Normal(0.0, 2.0).expand([K, n_features]).to_event(2),
        )

    asc_raw = pyro.sample(
        "asc_raw",
        dist.Normal(0.0, 1.0).expand([K, n_alts - 1]).to_event(2),
    )
    asc_full = torch.cat([torch.zeros(K, 1, dtype=X.dtype, device=X.device), asc_raw], dim=1)

    chosen_mask = choice > 0.5
    y_wide = torch.zeros(n_obs, dtype=torch.long, device=X.device)
    y_wide[obs_id[chosen_mask]] = alt_id[chosen_mask]

    scen_ll = torch.zeros(n_obs, K, dtype=X.dtype, device=X.device)
    for k in range(K):
        v_k = X @ beta[k] + asc_full[k][alt_id]
        v_wide_k = torch.full((n_obs, n_alts), -1e9, dtype=X.dtype, device=X.device)
        v_wide_k[obs_id, alt_id] = v_k
        scen_ll[:, k] = dist.Categorical(logits=v_wide_k).log_prob(y_wide)

    person_class_ll = torch.zeros(n_persons, K, dtype=X.dtype, device=X.device)
    person_class_ll.scatter_add_(0, person_obs_id.unsqueeze(1).expand_as(scen_ll), scen_ll)

    log_marginal = torch.logsumexp(log_class_probs + person_class_ll, dim=1)
    pyro.factor("loglik", log_marginal.sum())


# %%
def extract_point_estimates(posterior, K, prior_withconstraints):
    if prior_withconstraints:
        beta = -torch.nn.functional.softplus(posterior["beta_raw"])
    else:
        beta = posterior["beta"]

    asc_raw = posterior["asc_raw"]
    kappa = posterior["kappa"]
    lengthscale = posterior["lengthscale"]
    gp_latent = torch.stack([posterior[f"gp_latent_{k}"] for k in range(K)], dim=1)

    assert beta.shape == (K, 2)
    assert asc_raw.shape[0] == K
    assert kappa.shape[0] == K
    assert lengthscale.shape[0] == K
    assert gp_latent.shape[1] == K

    return (
        beta.detach().cpu().numpy(),
        asc_raw.detach().cpu().numpy(),
        kappa.detach().cpu().numpy(),
        lengthscale.detach().cpu().numpy(),
        gp_latent.detach().cpu().numpy(),
    )


def matern32_kernel_matrix(X1, X2, kappa, lengthscale):
    dist_matrix = torch.cdist(X1, X2)
    scaled = np.sqrt(3.0) * dist_matrix / lengthscale
    return (kappa**2) * (1.0 + scaled) * torch.exp(-scaled)


def gp_predict_latent(train_Z, train_f, target_Z, kappa, lengthscale):
    train_Z_t = torch.tensor(train_Z, dtype=torch.float32)
    train_f_t = torch.tensor(train_f, dtype=torch.float32)
    target_Z_t = torch.tensor(target_Z, dtype=torch.float32)
    kappa_t = torch.tensor(float(kappa), dtype=torch.float32)
    lengthscale_t = torch.tensor(float(lengthscale), dtype=torch.float32)

    k_tt = matern32_kernel_matrix(train_Z_t, train_Z_t, kappa_t, lengthscale_t)
    k_tt = k_tt + 1e-4 * torch.eye(k_tt.shape[0], dtype=torch.float32)
    k_ts = matern32_kernel_matrix(train_Z_t, target_Z_t, kappa_t, lengthscale_t)

    alpha = torch.linalg.solve(k_tt, train_f_t)
    pred_mean = k_ts.T @ alpha
    return pred_mean


def class_probs_from_gp(train_data, posterior, K, target_data=None):
    if target_data is None:
        target_data = train_data

    _, _, kappa, lengthscale, gp_latent = extract_point_estimates(posterior, K, prior_withconstraints=True)

    if target_data is train_data:
        logits = gp_latent
    else:
        logits_list = []
        for k in range(K):
            pred_mean = gp_predict_latent(
                train_data["Z"],
                gp_latent[:, k],
                target_data["Z"],
                kappa[k],
                lengthscale[k],
            )
            logits_list.append(pred_mean)
        logits = np.column_stack([col.detach().cpu().numpy() if torch.is_tensor(col) else col for col in logits_list])

    class_probs = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    class_share = class_probs.mean(axis=0)
    return class_probs, class_share


def mixture_prob_matrix(data, beta, asc_raw, class_probs):
    """Return mixture choice probabilities per observation and alternative."""
    n_obs = data["n_obs"]
    n_alts = data["n_alts"]
    K = beta.shape[0]

    X = np.column_stack([data["tt"], data["co"]])
    obs_id = data["obs_id"]
    alt_id = data["alt_id"]
    person_obs_id = data["person_obs_id"]

    asc_full = np.column_stack([np.zeros(K), asc_raw])

    class_choice_probs = np.zeros((n_obs, K, n_alts), dtype=float)

    for k in range(K):
        v_long = X @ beta[k] + asc_full[k][alt_id]
        v_wide = np.full((n_obs, n_alts), -1e9, dtype=float)
        v_wide[obs_id, alt_id] = v_long
        p_wide = np.exp(v_wide - logsumexp(v_wide, axis=1, keepdims=True))
        class_choice_probs[:, k, :] = p_wide

    mixture_probs = np.zeros((n_obs, n_alts), dtype=float)
    for t in range(n_obs):
        p_idx = person_obs_id[t]
        mixture_probs[t] = np.sum(
            class_probs[p_idx][:, None] * class_choice_probs[t],
            axis=0,
        )

    return np.clip(mixture_probs, 1e-12, 1.0)


def chosen_alt_per_obs(data):
    y_wide = np.zeros(data["n_obs"], dtype=int)
    chosen_mask = data["choice"] == 1
    y_wide[data["obs_id"][chosen_mask]] = data["alt_id"][chosen_mask]
    return y_wide


def point_estimate_ll_and_accuracy(data, beta, asc_raw, class_probs):
    probs = mixture_prob_matrix(data, beta, asc_raw, class_probs)
    y_true = chosen_alt_per_obs(data)

    chosen_p = probs[np.arange(len(y_true)), y_true]
    ll = float(np.log(chosen_p).sum())

    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())

    return ll, acc


def parameter_count(K, n_alts, n_Z):
    return K * 2 + K * (n_alts - 1) + 2 * K


def class_profile_table(beta, asc_raw, class_share, tt_std, co_std):
    """Build attribute-by-class table requested by user."""
    K = beta.shape[0]
    columns = [f"Class {k + 1}" for k in range(K)]

    asc_train = np.zeros(K)
    asc_car = asc_raw[:, 1]
    travel_time = beta[:, 0]
    travel_cost = beta[:, 1]

    vot = (travel_time / tt_std) / (travel_cost / co_std)

    profile = pd.DataFrame(
        {
            col: [
                asc_train[i],
                asc_car[i],
                travel_time[i],
                travel_cost[i],
                class_share[i],
                vot[i],
            ]
            for i, col in enumerate(columns)
        },
        index=[
            "ASC (Train)",
            "ASC (Car)",
            "Travel Time",
            "Travel Cost",
            "Class Share",
            "VOT (CHF/min)",
        ],
    )

    return profile


def interpret_best_model(K, profile_df):
    """Generate concise interpretation text for the selected best model."""
    class_cols = list(profile_df.columns)

    tt_vals = np.array([profile_df.loc["Travel Time", c] for c in class_cols])
    cost_vals = np.array([profile_df.loc["Travel Cost", c] for c in class_cols])
    asc_car_vals = np.array([profile_df.loc["ASC (Car)", c] for c in class_cols])

    tt_rank = np.argsort(tt_vals)  # more negative first
    cost_rank = np.argsort(cost_vals)  # more negative first
    car_rank = np.argsort(-asc_car_vals)  # higher first

    lines = []
    lines.append(f"Best model by predictive performance uses K={K} latent classes.")
    lines.append("Class interpretation:")

    for i, c in enumerate(class_cols):
        share = profile_df.loc["Class Share", c]
        tt = profile_df.loc["Travel Time", c]
        co = profile_df.loc["Travel Cost", c]
        asc_car = profile_df.loc["ASC (Car)", c]
        vot = profile_df.loc["VOT (CHF/min)", c]

        tt_pos = int(np.where(tt_rank == i)[0][0]) + 1
        co_pos = int(np.where(cost_rank == i)[0][0]) + 1
        car_pos = int(np.where(car_rank == i)[0][0]) + 1

        if asc_car > 0.25:
            car_pref = "car-preferring segment"
        elif asc_car < -0.25:
            car_pref = "non-car segment"
        else:
            car_pref = "neutral-to-car segment"

        stability_note = ""
        if share < 0.03:
            stability_note += " small-share class;"
        if abs(co) < 0.15:
            stability_note += " VOT may be unstable (cost coefficient near zero);"
        if stability_note:
            stability_note = f" [{stability_note.strip()}]"

        lines.append(
            (
                f"- {c}: share={share:.3f}, {car_pref}; "
                f"TT={tt:.3f} (time sensitivity rank {tt_pos}/{K}), "
                f"Cost={co:.3f} (cost sensitivity rank {co_pos}/{K}), "
                f"ASC(Car)={asc_car:.3f} (car preference rank {car_pos}/{K}), "
                f"VOT={vot:.3f} CHF/min.{stability_note}"
            )
        )

    lines.append("Interpretation note: more negative Travel Time/Travel Cost coefficients imply stronger disutility.")
    lines.append("VOT is computed as (dU/dTT)/(dU/dCost) after converting from scaled units to raw minutes and CHF.")

    return "\n".join(lines)


# %%
def fit_and_evaluate_k(train_data, test_data, K, prior_withconstraints=True, n_steps=250):
    train_torch = to_torch_lccm(train_data)

    pyro.clear_param_store()
    pyro.set_rng_seed(42 + K)

    def model_fn(
        X,
        obs_id,
        alt_id,
        choice,
        person_obs_id,
        Z,
        n_obs,
        n_alts,
        n_persons,
    ):
        return lccm_pyro_model(
            X,
            obs_id,
            alt_id,
            choice,
            person_obs_id,
            Z,
            n_obs,
            n_alts,
            n_persons,
            K=K,
            prior_withconstraints=prior_withconstraints,
        )

    guide = AutoNormal(model_fn)
    optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.9995})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    for step in range(n_steps):
        loss = svi.step(
            train_torch["X"],
            train_torch["obs_id"],
            train_torch["alt_id"],
            train_torch["choice"],
            train_torch["person_obs_id"],
            train_torch["Z"],
            train_torch["n_obs"],
            train_torch["n_alts"],
            train_torch["n_persons"],
        )
        if step % 100 == 0 or step == n_steps - 1:
            print(f"[K={K}] step {step:4d} | ELBO loss = {loss:,.2f}")

    posterior = guide.median(
        train_torch["X"],
        train_torch["obs_id"],
        train_torch["alt_id"],
        train_torch["choice"],
        train_torch["person_obs_id"],
        train_torch["Z"],
        train_torch["n_obs"],
        train_torch["n_alts"],
        train_torch["n_persons"],
    )

    beta, asc_raw, kappa, lengthscale, gp_latent = extract_point_estimates(posterior, K, prior_withconstraints)

    train_class_probs, class_share = class_probs_from_gp(train_data, posterior, K, target_data=train_data)
    test_class_probs, _ = class_probs_from_gp(train_data, posterior, K, target_data=test_data)

    train_ll, train_acc = point_estimate_ll_and_accuracy(train_data, beta, asc_raw, train_class_probs)
    test_ll, test_acc = point_estimate_ll_and_accuracy(test_data, beta, asc_raw, test_class_probs)

    pred_ll = test_ll

    n_params = parameter_count(K, train_data["n_alts"], train_data["Z"].shape[1])
    aic = 2 * n_params - 2 * train_ll
    bic = np.log(train_data["n_obs"]) * n_params - 2 * train_ll

    return {
        "K": K,
        "n_params": n_params,
        "log_likelihood": train_ll,
        "AIC": aic,
        "BIC": bic,
        "prediction_LL": pred_ll,
        "test_LL": test_ll,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "beta": beta,
        "asc_raw": asc_raw,
        "kappa": kappa,
        "lengthscale": lengthscale,
        "gp_latent": gp_latent,
        "class_share": class_share,
        "train_class_probs": train_class_probs,
        "test_class_probs": test_class_probs,
    }


# %%
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("swissmetro.csv")
    print(df.head())
    print(df.shape)

    valid_age = df["AGE"].notna() & (df["AGE"] != 6)
    valid_choice = df["CHOICE"] != 0
    valid_purpose = df["PURPOSE"] != 9
    df = df[valid_age & valid_choice & valid_purpose].copy()

    print("Filtered shape:", df.shape)

    unique_ids = df["ID"].unique()
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train_df = df[df["ID"].isin(train_ids)].copy()
    test_df = df[df["ID"].isin(test_ids)].copy()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train persons:", train_df["ID"].nunique())
    print("Test persons:", test_df["ID"].nunique())

    tt_cols = ["TRAIN_TT", "SM_TT", "CAR_TT"]
    co_cols = ["TRAIN_CO", "SM_CO", "CAR_CO"]

    tt_train_values = train_df[tt_cols].values.reshape(-1)
    co_train_values = train_df[co_cols].values.reshape(-1)

    tt_mean = tt_train_values.mean()
    tt_std = tt_train_values.std()
    co_mean = co_train_values.mean()
    co_std = co_train_values.std()

    for col in tt_cols:
        train_df[col + "_SCALED"] = (train_df[col] - tt_mean) / tt_std
        test_df[col + "_SCALED"] = (test_df[col] - tt_mean) / tt_std

    for col in co_cols:
        train_df[col + "_SCALED"] = (train_df[col] - co_mean) / co_std
        test_df[col + "_SCALED"] = (test_df[col] - co_mean) / co_std

    train_long_df = create_long_format(train_df)
    test_long_df = create_long_format(test_df)

    demo_cols = ["ID", "AGE", "MALE", "INCOME", "FIRST", "LUGGAGE", "PURPOSE", "GA"]
    person_demographic = df[demo_cols].drop_duplicates(subset="ID").copy()

    train_long_df = train_long_df.merge(person_demographic, on="ID", how="left")
    test_long_df = test_long_df.merge(person_demographic, on="ID", how="left")

    assert train_long_df[demo_cols].notna().all().all()
    assert test_long_df[demo_cols].notna().all().all()

    train_data = prepare_lccm_data(train_long_df, id_col="ID", include_GA=True)
    test_data = prepare_lccm_data(
        test_long_df,
        id_col="ID",
        include_GA=True,
        reference_Z_cols=train_data["Z_cols"],
    )

    rows = []
    profile_long_rows = []
    profile_tables = {}
    fit_results = {}

    for k in range(2, 10):
        print(f"\n--- Fitting K={k} ---")
        fit_result = fit_and_evaluate_k(
            train_data,
            test_data,
            K=k,
            prior_withconstraints=True,
        )
        fit_results[k] = fit_result

        profile_df = class_profile_table(
            fit_result["beta"],
            fit_result["asc_raw"],
            fit_result["class_share"],
            tt_std,
            co_std,
        )
        profile_tables[k] = profile_df

        profile_path = output_dir / f"gp_lccm_class_profile_K{k}.csv"
        profile_df.to_csv(profile_path, index=True)

        for attr in profile_df.index:
            for cls in profile_df.columns:
                profile_long_rows.append(
                    {
                        "K": k,
                        "attribute": attr,
                        "class": cls,
                        "value": float(profile_df.loc[attr, cls]),
                    }
                )

        row = {
            key: fit_result[key]
            for key in [
                "K",
                "n_params",
                "log_likelihood",
                "AIC",
                "BIC",
                "prediction_LL",
                "test_LL",
                "test_accuracy",
                "train_accuracy",
            ]
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

    print("\nModel comparison table (K=2..9):")
    print(
        results_df[
            [
                "K",
                "n_params",
                "log_likelihood",
                "AIC",
                "BIC",
                "prediction_LL",
                "test_LL",
                "test_accuracy",
            ]
        ].to_string(index=False)
    )

    comparison_csv = output_dir / "gp_lccm_comparison_k2_to_k9.csv"
    results_df.to_csv(comparison_csv, index=False)

    profile_long_df = pd.DataFrame(profile_long_rows)
    profile_long_csv = output_dir / "gp_lccm_class_profiles_long.csv"
    profile_long_df.to_csv(profile_long_csv, index=False)

    best_pred_idx = results_df["prediction_LL"].idxmax()
    best_acc_idx = results_df["test_accuracy"].idxmax()
    best_k = int(results_df.loc[best_pred_idx, "K"])

    print("\nBest by prediction_LL:")
    print(results_df.loc[[best_pred_idx], ["K", "prediction_LL", "test_accuracy"]].to_string(index=False))

    print("\nBest by test_accuracy:")
    print(results_df.loc[[best_acc_idx], ["K", "prediction_LL", "test_accuracy"]].to_string(index=False))

    best_profile_df = profile_tables[best_k]
    best_profile_path = output_dir / f"gp_lccm_best_model_profile_K{best_k}.csv"
    best_profile_df.to_csv(best_profile_path, index=True)

    best_fit = fit_results[best_k]
    person_prob_rows = []

    for dataset_name, dataset_data, probs in [
        ("train", train_data, best_fit["train_class_probs"]),
        ("test", test_data, best_fit["test_class_probs"]),
    ]:
        for idx, person_id in enumerate(dataset_data["person_ids"]):
            row = {
                "dataset": dataset_name,
                "ID": person_id,
                "predicted_class": int(np.argmax(probs[idx]) + 1),
            }
            for k in range(probs.shape[1]):
                row[f"class_{k + 1}_prob"] = float(probs[idx, k])
            person_prob_rows.append(row)

    person_prob_df = pd.DataFrame(person_prob_rows)
    person_prob_csv = output_dir / f"gp_lccm_person_class_probs_bestK{best_k}.csv"
    person_prob_df.to_csv(person_prob_csv, index=False)

    interpretation_text = interpret_best_model(best_k, best_profile_df)
    interpretation_path = output_dir / f"gp_lccm_best_model_interpretation_K{best_k}.txt"
    interpretation_path.write_text(interpretation_text)

    print("\nBest model interpretation:")
    print(interpretation_text)

    print("\nSaved files:")
    print(f"- {comparison_csv}")
    print(f"- {profile_long_csv}")
    print(f"- {best_profile_path}")
    print(f"- {person_prob_csv}")
    print(f"- {interpretation_path}")


if __name__ == "__main__":
    main()
