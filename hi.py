from pyro.ops.indexing import Vindex

def model(X_long, alt_id, obs_id, choice_long, person_id, n_alts=3, K=5):
    """
    Bayesian Latent Class Multinomial Logit (LCCM)
    - X_long:     (N_long, n_features)  ← tt, cost (standardized)
    - alt_id:     (N_long,)             ← 0=Train, 1=SwissMetro, 2=Car
    - obs_id:     (N_long,)             ← choice situation ID
    - choice_long:(N_long,)             ← 0 or 1 (chosen)
    - person_id:  (N_long,)             ← person ID (important!)
    """
    N_long = X_long.shape[0]
    n_persons = len(torch.unique(person_id))

    # === Priors on class-specific parameters ===
    with pyro.plate("classes", K):
        beta = pyro.sample("beta", dist.Normal(0., 5.).expand([X_long.shape[1]]).to_event(1))
        asc_free = pyro.sample("asc", dist.Normal(0., 5.).expand([n_alts-1]).to_event(1))
        asc = torch.cat([torch.zeros(K, 1), asc_free], dim=1)   # fix ASC for alt 0 = 0

    # === Class assignment per person (shared across all observations of that person) ===
    with pyro.plate("persons", n_persons):
        q = pyro.sample("q", dist.Categorical(torch.ones(K) / K))   # q.shape = (n_persons,)

    # === Get class for every long-format row ===
    q_long = q[person_id]                     # (N_long,)

    # === Class-specific parameters for each row ===
    beta_long = Vindex(beta)[..., q_long, :]   # (N_long, n_features)
    asc_long  = Vindex(asc)[..., q_long, :]    # (N_long, n_alts)

    # === Linear predictor per alternative ===
    linear = (X_long * beta_long).sum(dim=-1)          # (N_long,)

    # Add the correct ASC for this alternative
    V_row = linear + asc_long[torch.arange(N_long), alt_id]   # (N_long,)

    # === Likelihood: one choice per observation (obs_id) ===
    with pyro.plate("observations", N_long):
        # We use the long-format Bernoulli trick for MNL
        # (works very well and is numerically stable)
        pyro.sample("choice", dist.Bernoulli(logits=V_row), obs=choice_long)