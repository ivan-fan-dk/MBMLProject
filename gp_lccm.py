# %%
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
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

    gamma = pyro.sample(
        "gamma",
        dist.Normal(0.0, 1.5).expand([K - 1, Z.shape[1] + 1]).to_event(2),
    )

    Z_aug = torch.cat(
        [torch.ones(n_persons, 1, dtype=Z.dtype, device=Z.device), Z],
        dim=1,
    )
    eta_nonbase = Z_aug @ gamma.T
    eta = torch.cat(
        [eta_nonbase, torch.zeros(n_persons, 1, dtype=Z.dtype, device=Z.device)],
        dim=1,
    )
    log_class_probs = torch.log_softmax(eta, dim=1)

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
    gamma = posterior["gamma"]

    assert beta.shape == (K, 2)
    assert asc_raw.shape[0] == K
    assert gamma.shape[0] == K - 1

    return beta.detach().cpu().numpy(), asc_raw.detach().cpu().numpy(), gamma.detach().cpu().numpy()


def mixture_prob_matrix(data, beta, asc_raw, gamma):
    """Return mixture choice probabilities per observation and alternative."""
    n_obs = data["n_obs"]
    n_alts = data["n_alts"]
    n_persons = data["n_persons"]
    K = beta.shape[0]

    X = np.column_stack([data["tt"], data["co"]])
    obs_id = data["obs_id"]
    alt_id = data["alt_id"]
    person_obs_id = data["person_obs_id"]

    asc_full = np.column_stack([np.zeros(K), asc_raw])

    Z_aug = np.column_stack([np.ones(n_persons), data["Z"]])
    eta_nonbase = Z_aug @ gamma.T
    eta = np.column_stack([eta_nonbase, np.zeros(n_persons)])
    class_probs = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))

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


def point_estimate_ll_and_accuracy(data, beta, asc_raw, gamma):
    probs = mixture_prob_matrix(data, beta, asc_raw, gamma)
    y_true = chosen_alt_per_obs(data)

    chosen_p = probs[np.arange(len(y_true)), y_true]
    ll = float(np.log(chosen_p).sum())

    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())

    return ll, acc


def parameter_count(K, n_alts, n_Z):
    return K * 2 + K * (n_alts - 1) + (K - 1) * (n_Z + 1)


def predictive_ll_from_guide(guide, train_torch, test_data, K, prior_withconstraints, n_draws=80):
    """Posterior predictive LL on test by log-mean-exp over guide draws."""
    draw_ll = []

    for _ in range(n_draws):
        sample = guide(
            train_torch["X"],
            train_torch["obs_id"],
            train_torch["alt_id"],
            train_torch["choice"],
            train_torch["person_obs_id"],
            train_torch["Z"],
            train_torch["n_obs"],
            train_torch["n_alts"],
            train_torch["n_persons"],
            K=K,
            prior_withconstraints=prior_withconstraints,
        )

        beta, asc_raw, gamma = extract_point_estimates(sample, K, prior_withconstraints)
        ll_i, _ = point_estimate_ll_and_accuracy(test_data, beta, asc_raw, gamma)
        draw_ll.append(ll_i)

    draw_ll = np.asarray(draw_ll)
    return float(logsumexp(draw_ll) - np.log(len(draw_ll)))


# %%
def fit_and_evaluate_k(train_data, test_data, K, prior_withconstraints=True, n_steps=900):
    train_torch = to_torch_lccm(train_data)

    pyro.clear_param_store()
    pyro.set_rng_seed(42 + K)

    guide = AutoNormal(lccm_pyro_model)
    optimizer = ClippedAdam({"lr": 0.03, "lrd": 0.9995})
    svi = SVI(lccm_pyro_model, guide, optimizer, loss=Trace_ELBO())

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
            K=K,
            prior_withconstraints=prior_withconstraints,
        )
        if step % 300 == 0 or step == n_steps - 1:
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
        K=K,
        prior_withconstraints=prior_withconstraints,
    )

    beta, asc_raw, gamma = extract_point_estimates(posterior, K, prior_withconstraints)

    train_ll, train_acc = point_estimate_ll_and_accuracy(train_data, beta, asc_raw, gamma)
    test_ll, test_acc = point_estimate_ll_and_accuracy(test_data, beta, asc_raw, gamma)

    pred_ll = predictive_ll_from_guide(
        guide,
        train_torch,
        test_data,
        K,
        prior_withconstraints,
        n_draws=80,
    )

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
    }


# %%
def main():
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
    for k in range(2, 10):
        print(f"\n--- Fitting K={k} ---")
        row = fit_and_evaluate_k(
            train_data,
            test_data,
            K=k,
            prior_withconstraints=True,
            n_steps=900,
        )
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

    best_pred_idx = results_df["prediction_LL"].idxmax()
    best_acc_idx = results_df["test_accuracy"].idxmax()

    print("\nBest by prediction_LL:")
    print(results_df.loc[[best_pred_idx], ["K", "prediction_LL", "test_accuracy"]].to_string(index=False))

    print("\nBest by test_accuracy:")
    print(results_df.loc[[best_acc_idx], ["K", "prediction_LL", "test_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
