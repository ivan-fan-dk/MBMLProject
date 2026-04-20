from pyro.ops.indexing import Vindex

def model(X_obs, Y_obs, obs_id, alt_id, n_alts=3):
    """
    Bayesian Multinomial Logit Model in Pyro
    
    Parameters:
    - X_obs: (n_obs, n_features) alternative attributes in long format
    - Y_obs: (n_obs, 1) choice indicator in long format (1 if chosen, 0 otherwise)
    - beta: (n_classes, n_features) parameter multiplier
    - asc: (n_classes, n_alts) 
    - q: (n_person, n_classes) contains binary variables. q_{i,j} indicates whether i'th person belong to j class
    
    - obs_id: observation ID for grouping alternatives
    - alt_id: alternative ID (0, 1, 2)
    - n_alts: number of alternatives (3: Train, SwissMetro, Car)
    - V: (N_alts,) linear utility function: V_i = X_i @ beta + ASC_i
    """
    n_obs, n_features = X_obs.shape
    n_ids = len(obs_id.unique())
    K = 5

    # Priors on coefficients: explicitly expand to vector
    with pyro.plate('Number of classes', K):
        # (n_classes, n_features)
        betas = pyro.sample('beta', dist.MultivariateNormal(torch.zeros(n_features), torch.eye(n_features)))
    
        # Alternative-specific constants: fix first to 0 for identification
        # Sample only for alternatives 1 and 2
        # (K, n_alts-1)
        ascs_free = pyro.sample('asc', dist.MultivariateNormal(torch.zeros(n_alts - 1), torch.eye(n_alts - 1)))

    # (K, n_alts): fix first ASC to 0 for identification
    zero_col = torch.zeros(K, 1)
    ascs = torch.cat([zero_col, ascs_free], dim=-1)
    
    # Likelihood: categorical (multinomial logit)
    with pyro.plate('Number of people', n_ids):
        # (n_obs, 1)
        Q = pyro.sample("q", dist.Categorical(torch.ones(K)/K)) # TODO obs is needed here.
    
    Q_obs = Q[..., obs_id]
    
    # Shape: (..., n_obs, n_features)
    beta_obs = Vindex(betas)[..., Q_obs, :]
    
    # Shape: (..., n_obs, n_alts)
    asc_obs = Vindex(ascs)[..., Q_obs, :]

    # (n_obs,) <- (n_obs, n_features) * (n_obs, n_features)
    X_deterministic = y = pyro.deterministic("X", X_obs)
    
    U = torch.sum(X_deterministic * beta_obs, dim=-1)
    
    # (n_obs, n_alts) <- (n_obs, n_alts) + (n_obs, 1)
    V = asc_obs + U[:, None]
    
    with pyro.plate('Number of observations', n_obs):
        y = pyro.sample('y', dist.Categorical(logits=V[obs_id]), obs=Y_obs)

pyro.render_model(model, model_args=(X, Y, obs_id, alt_id), render_distributions=True, render_deterministic=True, render_params=True)