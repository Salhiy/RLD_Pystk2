from .actors import SquashedGaussianActor, HistoryWrapper

def init_env(cfg, autoreset, use_ai=True):
	train_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, autoreset=autoreset, wrappers=[HistoryWrapper], agent=AgentSpec(use_ai=True)),
        cfg.algorithm.n_envs,
        include_last_state=include_last_state
    ).seed(cfg.algorithm.seed)

    tr_agent = Agents(train_env_agent, actor)
    train_agent = TemporalAgent(tr_agent)
    train_workspace = Workspace()

    return train_agent, train_workspace

def imitation_learning(cfg, actor):

	#env with the bot
	train_agent, train_workspace = init_env(cfg, autoreset=True)

    for episode in cfg.algorithm.imitation_learning_steps:
    	train_agent(
                train_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=False,
            )

    observations = {
    	"discrete" : train_workspace["env/env_obs/discrete"],
    	"continuous" : train_workspace["env/env_obs/continuous"]
    }
    actions = {
    	"discrete" : train_workspace["action/discrete"],
    	"continuous" : train_workspace["action/continuous"]
    }
    #start the learning
    num_obs = len(observations["discrete"])
	# Définition des pertes et de l'optimiseur
	criterion_discrete = nn.CrossEntropyLoss()  
	criterion_continuous = nn.MSELoss()  
	optimizer = optim.Adam(actor.parameters(), lr=0.001)

	# Entraînement
	num_epochs = cfg.algorithm.imitation_learning_steps
	batch_size = cfg.algorithm.batch_size
	train_agent, train_workspace_v2 = init_env(cfg, autoreset=False)

	for epoch in num_epochs:
		for i in range(0, num_obs):

		    optimizer.zero_grad()

		    # Prédiction du modèle
		    train_workspace_v2["env/env_obs/discrete"].set_full(observations["discrete"])
		    train_workspace_v2["env/env_obs/continuous"].set_full(observations["continuous"])
		    
		    train_agent(
		    	train_workspace_v2,
                t=i,
                nb_steps=1,
                stochastic=False
		    )

		    action_discrete = train_workspace_v2["action/discrete"][i]
		    action_continious = train_workspace_v2["action/continuous"][i]

		    loss_discrete = criterion_discrete(action_discrete, actions["discrete"][i])
		    loss_continuous = criterion_continuous(action_continious, actions["continuous"][i])
		   
		    # Somme des pertes (on peut aussi les pondérer)
		    loss = loss_discrete + loss_continuous

		    # Rétropropagation et optimisation
		    loss.backward()
		    optimizer.step()

	    if epoch % 5 == 0:
	        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
