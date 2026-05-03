# %%
import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split


# Set seeds for reproducibility
np.random.seed(11)
torch.manual_seed(11)
pyro.set_rng_seed(11)

# %% [markdown]
# train test splitting

# %%
df = pd.read_csv('swissmetro.csv')
print(df.head())
print(df.shape)

# Keep only observations with known age, explicit choices, and non-'other' trip purposes.
# In this file, AGE==6 is the unknown age code and PURPOSE==9 is 'other'.
valid_age = df['AGE'].notna() & (df['AGE'] != 6)
valid_choice = df['CHOICE'] != 0
valid_purpose = df['PURPOSE'] != 9
df = df[valid_age & valid_choice & valid_purpose].copy()

print("Filtered shape:", df.shape)

# -----------------------------
# Train / Test split
# -----------------------------
# IMPORTANT:
# split by individual ID, not by row
# to avoid the same person appearing in both train and test
unique_ids = df["ID"].unique()

train_ids, test_ids = train_test_split(
    unique_ids,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_df = df[df["ID"].isin(train_ids)].copy()
test_df = df[df["ID"].isin(test_ids)].copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("Train persons:", train_df["ID"].nunique())
print("Test persons:", test_df["ID"].nunique())


# -----------------------------
# Attribute-level scaling
# -----------------------------
# One scaling rule for all travel time variables
# One scaling rule for all travel cost variables
# Fit on train only, apply to both train and test
# As long as our utility function is like this: U = beta_tt * tt + beta_co * co + ASC

tt_cols = ["TRAIN_TT", "SM_TT", "CAR_TT"]
co_cols = ["TRAIN_CO", "SM_CO", "CAR_CO"]

# calculate train-only scaling factors
tt_train_values = train_df[tt_cols].values.reshape(-1)
co_train_values = train_df[co_cols].values.reshape(-1)

tt_mean = tt_train_values.mean()
tt_std = tt_train_values.std()

co_mean = co_train_values.mean()
co_std = co_train_values.std()

# apply same TT scaling to all TT columns
for col in tt_cols:
    train_df[col + "_SCALED"] = (train_df[col] - tt_mean) / tt_std
    test_df[col + "_SCALED"] = (test_df[col] - tt_mean) / tt_std

# apply same CO scaling to all CO columns
for col in co_cols:
    train_df[col + "_SCALED"] = (train_df[col] - co_mean) / co_std
    test_df[col + "_SCALED"] = (test_df[col] - co_mean) / co_std

# Create long-format data for MNL estimation
# Each row observation becomes 3 rows (one per alternative that is available)
def create_long_format(df):
    long_data = []

    # Create a mapping from original indices to sequential obs_id (0, 1, 2, ...)
    obs_id_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(df.index.unique())}

    for idx, row in df.iterrows():
        obs_id_sequential = obs_id_map[idx]  # Map to sequential ID
        original_id = row['ID']  # Keep track of the individual ID
        
        # Train (alt_id 0)
        if row['TRAIN_AV'] == 1:
            long_data.append({
                'obs_id': obs_id_sequential,
                'ID': original_id,
                'alt_id': 0,
                'tt': row['TRAIN_TT_SCALED'],
                'co': row['TRAIN_CO_SCALED'],
                'choice': 1 if row['CHOICE'] == 1 else 0
            })
        
        # SwissMetro (alt_id 1)
        if row['SM_AV'] == 1:
            long_data.append({
                'obs_id': obs_id_sequential,
                'ID': original_id,
                'alt_id': 1,
                'tt': row['SM_TT_SCALED'],
                'co': row['SM_CO_SCALED'],
                'choice': 1 if row['CHOICE'] == 2 else 0
            })
        
        # Car (alt_id 2)
        if row['CAR_AV'] == 1:
            long_data.append({
                'obs_id': obs_id_sequential,
                'ID': original_id,
                'alt_id': 2,
                'tt': row['CAR_TT_SCALED'],
                'co': row['CAR_CO_SCALED'],
                'choice': 1 if row['CHOICE'] == 3 else 0
            })

    long_df = pd.DataFrame(long_data)

    # Validate: exactly one choice per obs_id
    choices_per_obs = long_df.groupby('obs_id')['choice'].sum()
    assert (choices_per_obs == 1).all(), "Error: some observations have != 1 choice"
    print("✓ Validation passed: exactly one choice per observation")

    print(f"\nLong format shape: {long_df.shape}")
    print(f"Number of unique observations: {long_df['obs_id'].nunique()}")
    print(f"Number of unique individuals: {long_df['ID'].nunique()}")
    print(f"Obs_id range: [{long_df['obs_id'].min()}, {long_df['obs_id'].max()}]")
    print("\nSample (first 9 rows = 3 obs):")
    long_df.head(9)
    return long_df

train_long_df = create_long_format(train_df)
test_long_df = create_long_format(test_df)

test_long_df.head(9)

# %%
train_long_df

# %%
def prepare_lccm_data(long_df, id_col="ID", include_GA=False, reference_Z_cols=None):
    def recode_membership_variables(person_df):
        person_df = person_df.copy()

        # Paper-style income recoding:
        # 0 or 1: under 50
        # 2: 50-100
        # 3: over 100
        # 4: unknown
        income_map = {
            0: 1,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }
        person_df["INCOME_REC"] = person_df["INCOME"].map(income_map)

        # Paper-style luggage recoding:
        # 0: none
        # 1: one piece
        # 3: several pieces -> 2: more than one piece
        luggage_map = {
            0: 0,
            1: 1,
            3: 2,
        }
        person_df["LUGGAGE_REC"] = person_df["LUGGAGE"].map(luggage_map)

        # Paper-style purpose recoding:
        # 1/5: commuter
        # 2/6: shopping
        # 3/7: business
        # 4/8: leisure
        purpose_map = {
            1: 1, 5: 1,
            2: 2, 6: 2,
            3: 3, 7: 3,
            4: 4, 8: 4,
        }
        person_df["PURPOSE_REC"] = person_df["PURPOSE"].map(purpose_map)

        return person_df

    def build_membership_matrix(person_df, include_GA=False, reference_cols=None):
        """Basically one-hot encoding of all the person-level variables, with some recoding and the option to drop one category per variable as reference."""
        person_df = recode_membership_variables(person_df)

        binary_vars = ["MALE", "FIRST"]
        if include_GA:
            binary_vars.append("GA")

        Z = person_df[binary_vars].astype(float).copy()

        cat_vars = ["AGE", "INCOME_REC", "LUGGAGE_REC", "PURPOSE_REC"]

        for var in cat_vars:
            dummies = pd.get_dummies(person_df[var], prefix=var, dtype=float)

            # drop first category as base
            dummies = dummies.iloc[:, 1:]

            Z = pd.concat([Z, dummies], axis=1)
            # print(f"Added {dummies.shape[1]} columns for {var}, now Z has shape {Z.shape}")

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

    person_df = (
        df.groupby(id_col, as_index=False)
        .first()[demo_cols]
        .copy()
    )

    person_df["person_idx"] = person_df[id_col].map(
        dict(zip(obs_person[id_col], obs_person["person_idx"]))
    )

    person_df = person_df.sort_values("person_idx").reset_index(drop=True)

    Z_df = build_membership_matrix(
        person_df,
        include_GA=include_GA,
        reference_cols=reference_Z_cols
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

id_col = "ID"
demo_cols = [id_col, "AGE", "MALE", "INCOME", "FIRST", "LUGGAGE", "PURPOSE", "GA"]

person_demograhic = (
    df[demo_cols]
    .drop_duplicates(subset=id_col)
    .copy()
)

train_df = train_long_df.merge(person_demograhic, on=id_col, how="left")
test_df = test_long_df.merge(person_demograhic, on=id_col, how="left")

assert train_df[demo_cols].notna().all().all()
assert test_df[demo_cols].notna().all().all()

train_data = prepare_lccm_data(
    train_df,
    id_col=id_col,
)

test_data = prepare_lccm_data(
    test_df,
    id_col=id_col,
    reference_Z_cols=train_data["Z_cols"]
)

# %%
# train_data["Z"], row is person, column is membership variable ["MALE", "FIRST", "AGE"(4), "INCOME_REC"(3), "LUGGAGE_REC"(2), "PURPOSE_REC"(3)]. Note that reference is omitted for each variable.
train_data.keys(), train_data["Z_cols"], train_data["Z"].shape, train_data['tt'].shape

# %%
train_data["Z"].shape

# %%
# in the code below tau represents the distance between to input points, i.e. tau = ||x_n - x_m||.
def squared_exponential(x1, x2, kappa, lengthscale):
    tau = np.linalg.norm(x1 - x2)
    return kappa**2*np.exp(-0.5*tau**2/lengthscale**2)

def matern12(x1, x2, kappa, lengthscale):
    tau = np.linalg.norm(x1 - x2)
    return kappa**2*np.exp(-tau/lengthscale)

def matern32(x1, x2, kappa, lengthscale):
    tau = np.linalg.norm(x1 - x2)
    return kappa**2*(1 + np.sqrt(3)*tau/lengthscale)*np.exp(-np.sqrt(3)*tau/lengthscale)

def lccm_pyro_model(X, obs_id, alt_id, choice, person_obs_id, Z, n_obs, n_alts=3, n_persons=None, K=2, prior_withconstraints=False):
    """Runnable latent class choice model with person-level membership covariates."""
    n_persons = int(person_obs_id.max().item()) + 1
    n_features = X.shape[1]

    Z_aug = torch.cat(
        [torch.ones(n_persons, 1, dtype=Z.dtype, device=X.device), Z],
        dim=1,
    )

    gamma = pyro.sample(
        "gamma",
        dist.Normal(0.0, 1.0).expand([K - 1, Z_aug.shape[1]]).to_event(2),
    )

    eta_nonbase = Z_aug @ gamma.T
    eta = torch.cat(
        [eta_nonbase, torch.zeros(n_persons, 1, dtype=Z.dtype, device=X.device)],
        dim=1,
    )
    log_class_probs = torch.log_softmax(eta, dim=1)

    if prior_withconstraints is False:
        beta = pyro.sample(
            "beta",
            dist.Normal(0.0, 2.0).expand([K, n_features]).to_event(2),
        )
    else:
        beta_raw = pyro.sample(
            "beta_raw",
            dist.Normal(0.0, 1.0).expand([K, n_features]).to_event(2),
        )
        beta = -torch.nn.functional.softplus(beta_raw)

    asc_raw = pyro.sample(
        "asc_raw",
        dist.Normal(0.0, 1.0).expand([K, n_alts - 1]).to_event(2),
    )
    asc_full = torch.cat([torch.zeros(K, 1, dtype=X.dtype, device=X.device), asc_raw], dim=1)

    chosen_mask = choice > 0.5
    Y_wide = torch.zeros(n_obs, dtype=torch.long, device=X.device)
    Y_wide[obs_id[chosen_mask]] = alt_id[chosen_mask]

    scen_ll_list = []
    for k in range(K):
        V_k = X @ beta[k] + asc_full[k][alt_id]
        V_wide_k = torch.full((n_obs, n_alts), -1e9, dtype=X.dtype, device=X.device)
        V_wide_k[obs_id, alt_id] = V_k
        scen_ll_list.append(dist.Categorical(logits=V_wide_k).log_prob(Y_wide))

    scen_ll = torch.stack(scen_ll_list, dim=1)

    person_class_ll = torch.zeros(n_persons, K, dtype=X.dtype, device=X.device)
    person_class_ll.scatter_add_(0, person_obs_id.unsqueeze(1).expand_as(scen_ll), scen_ll)

    log_marginal = torch.logsumexp(log_class_probs + person_class_ll, dim=1)
    pyro.factor("loglik", log_marginal.sum())

# %%
def to_torch_lccm(data):
    return {
        "X": torch.tensor(
            np.column_stack([data["tt"], data["co"]]),
            dtype=torch.float32
        ),
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
# -----------------------------
# 4. Log-likelihood
# -----------------------------
def log_likelihood_lccm(params, data, K):
    n_alts = data["n_alts"]
    n_Z = data["Z"].shape[1]
    n_persons = data["n_persons"]
    n_obs = data["n_obs"]

    idx = 0
    beta_tt = params[idx: idx + K]
    idx += K
    beta_co = params[idx: idx + K]
    idx += K
    asc = params[idx: idx + K * (n_alts - 1)].reshape(K, n_alts - 1)
    idx += K * (n_alts - 1)
    gamma = params[idx:].reshape(K - 1, n_Z + 1)

    X = np.column_stack([data["tt"], data["co"]])
    beta = np.column_stack([beta_tt, beta_co])
    Z_aug = np.column_stack([np.ones(n_persons), data["Z"]])
    eta_nonbase = Z_aug @ gamma.T
    eta = np.column_stack([eta_nonbase, np.zeros(n_persons)])
    log_class_probs = eta - logsumexp(eta, axis=1, keepdims=True)

    chosen_mask = data["choice"] > 0.5
    Y_wide = np.zeros(n_obs, dtype=int)
    Y_wide[data["obs_id"][chosen_mask]] = data["alt_id"][chosen_mask]

    person_class_ll = np.zeros((n_persons, K))
    for k in range(K):
        V_k = X @ beta[k] + asc[k][data["alt_id"]]
        V_wide_k = np.full((n_obs, n_alts), -1e9)
        V_wide_k[data["obs_id"], data["alt_id"]] = V_k
        scen_ll = dist.Categorical(
            logits=torch.tensor(V_wide_k, dtype=torch.float32)
        ).log_prob(torch.tensor(Y_wide, dtype=torch.long)).numpy()
        np.add.at(person_class_ll[:, k], data["person_obs_id"], scen_ll)

    log_marginal = logsumexp(log_class_probs + person_class_ll, axis=1)
    return float(log_marginal.sum())

# %%
def run_gp_lccm(K_b, df, long_df_train, long_df_test, prior_withconstraints):
    # -----------------------------
    # Prepare data for Bayesian Model
    # -----------------------------
    demo_cols_b = ["ID", "AGE", "MALE", "INCOME", "FIRST", "LUGGAGE", "PURPOSE", "GA"]

    person_demo_b = (
        df[demo_cols_b]
        .drop_duplicates(subset="ID")
        .copy()
    )

    long_df_train_b = long_df_train.merge(person_demo_b, on="ID", how="left")
    long_df_test_b = long_df_test.merge(person_demo_b, on="ID", how="left")

    assert long_df_train_b[demo_cols_b].notna().all().all()
    assert long_df_test_b[demo_cols_b].notna().all().all()

    train_b = prepare_lccm_data(long_df_train_b, id_col="ID", include_GA=True)

    test_b = prepare_lccm_data(
        long_df_test_b,
        id_col="ID",
        include_GA=True,
        reference_Z_cols=train_b["Z_cols"]
    )

    assert train_b["n_alts"] == test_b["n_alts"]
    assert train_b["Z"].shape[1] == test_b["Z"].shape[1]

    train_torch_b = to_torch_lccm(train_b)
    test_torch_b = to_torch_lccm(test_b)

    pyro.clear_param_store()
    pyro.set_rng_seed(42)

    guide_b = AutoNormal(lccm_pyro_model)

    optimizer_b = ClippedAdam({"lr": 0.03, "lrd": 0.9995})

    svi_b = SVI(
        lccm_pyro_model,
        guide_b,
        optimizer_b,
        loss=Trace_ELBO()
    )

    losses_b = []
    n_steps = 500

    for step in range(n_steps):
        loss = svi_b.step(
            train_torch_b["X"],
            train_torch_b["obs_id"],
            train_torch_b["alt_id"],
            train_torch_b["choice"],
            train_torch_b["person_obs_id"],
            train_torch_b["Z"],
            train_torch_b["n_obs"],
            train_torch_b["n_alts"],
            train_torch_b["n_persons"],
            K=K_b,
            prior_withconstraints=prior_withconstraints
        )

        losses_b.append(loss)

        if step % 500 == 0 or step == n_steps - 1:
            print(f"step {step:4d} | ELBO loss = {loss:,.2f}")

    posterior_b = guide_b.median(
        train_torch_b["X"],
        train_torch_b["obs_id"],
        train_torch_b["alt_id"],
        train_torch_b["choice"],
        train_torch_b["person_obs_id"],
        train_torch_b["Z"],
        train_torch_b["n_obs"],
        train_torch_b["n_alts"],
        train_torch_b["n_persons"],
        K=K_b,
        prior_withconstraints=prior_withconstraints
    )

    if prior_withconstraints is False:
        beta_b = posterior_b["beta"]
    else:
        beta_raw_b = posterior_b["beta_raw"]
        beta_b = -torch.nn.functional.softplus(beta_raw_b)

    asc_b = posterior_b["asc_raw"]
    gamma_b = posterior_b["gamma"]

    # -----------------------------
    # Choice coefficient table
    # -----------------------------
    choice_rows_b = []

    for k in range(K_b):
        choice_rows_b.append({
            "class": k + 1,
            "parameter": "beta_tt",
            "estimate": beta_b[k, 0].item()
        })
        choice_rows_b.append({
            "class": k + 1,
            "parameter": "beta_co",
            "estimate": beta_b[k, 1].item()
        })

        for j in range(1, train_torch_b["n_alts"]):
            choice_rows_b.append({
                "class": k + 1,
                "parameter": f"ASC_alt_{j}",
                "estimate": asc_b[k, j - 1].item()
            })

    choice_coef_b = pd.DataFrame(choice_rows_b)

    # -----------------------------
    # Membership coefficient table
    # -----------------------------
    membership_cols_b = ["intercept"] + train_torch_b["Z_cols"]

    membership_dict = {"variable": membership_cols_b}

    for k in range(K_b - 1):
        membership_dict[f"gamma_class_{k+1}_vs_class_{K_b}"] = (
            gamma_b[k].detach().numpy()
        )

    membership_coef_b = pd.DataFrame(membership_dict)

    print("\nBayesian Model — choice coefficients:")
    print(choice_coef_b.to_string(index=False))

    print("\nBayesian Model — membership coefficients:")
    print(membership_coef_b.to_string(index=False))

    plt.figure(figsize=(10, 3))
    plt.plot(losses_b)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.title("Bayesian Model: SVI convergence")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Posterior average class shares
    # -----------------------------
    Z_train = train_torch_b["Z"]
    n_persons = train_torch_b["n_persons"]

    Z_aug = torch.cat([
        torch.ones(n_persons, 1, dtype=Z_train.dtype, device=Z_train.device),
        Z_train
    ], dim=1)

    eta_nonbase = Z_aug @ gamma_b.T
    eta = torch.cat([
        eta_nonbase,
        torch.zeros(n_persons, 1, dtype=Z_train.dtype, device=Z_train.device)
    ], dim=1)

    class_probs = torch.softmax(eta, dim=1)
    avg_class_share = class_probs.mean(dim=0)

    print("\nPosterior average class shares:")
    for k in range(K_b):
        print(f"Class {k+1}: {avg_class_share[k].item():.4f}")

    class_share_b = avg_class_share.detach().numpy()

    # -----------------------------
    # Convert posterior median to numpy
    # Order expected by log_likelihood_lccm:
    # beta_tt, beta_co, asc, gamma
    # -----------------------------
    beta_tt_np = beta_b[:, 0].detach().numpy()
    beta_co_np = beta_b[:, 1].detach().numpy()

    params_b = np.concatenate([
        beta_tt_np,
        beta_co_np,
        asc_b.detach().numpy().flatten(),
        gamma_b.detach().numpy().flatten()
    ])

    # -----------------------------
    # Compute train/test LL
    # -----------------------------
    # train_ll_b = log_likelihood_lccm(params_b, train_b, K=K_b)
    # test_ll_b = log_likelihood_lccm(params_b, test_b, K=K_b)

    n_Z = train_b["Z"].shape[1]
    n_alts = train_b["n_alts"]

    n_params_b = (
        K_b
        + K_b
        + K_b * (n_alts - 1)
        + (K_b - 1) * (n_Z + 1)
    )

    summary_b = {
        "K": K_b,
        "prior_withconstraints": prior_withconstraints,
        "include_GA": True,
        "n_params": n_params_b,
        # "train_ll": train_ll_b,
        # "test_ll": test_ll_b,
        "final_elbo_loss": losses_b[-1],
    }

    print("\nBayesian Model predictive log-likelihood")
    # print(f"Train LL: {train_ll_b:.3f}")
    # print(f"Test  LL: {test_ll_b:.3f}")

    print("\nBayesian model summary:")
    print(pd.DataFrame([summary_b]).to_string(index=False))

    return {
        "summary": summary_b,
        "params_hat": params_b,
        "posterior": posterior_b,
        "choice_coef_table": choice_coef_b,
        "membership_table": membership_coef_b,
        "class_share": class_share_b,
        "losses": losses_b,
        "train_data": train_b,
        "test_data": test_b,
        "train_torch": train_torch_b,
        "test_torch": test_torch_b,
        "guide": guide_b,
    }

# %%
model2classes = run_gp_lccm(K_b=2, df=df, long_df_train=train_long_df, long_df_test=test_long_df, prior_withconstraints=True)

# %%



