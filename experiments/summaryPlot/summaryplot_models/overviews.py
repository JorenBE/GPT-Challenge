prop_results_dict = {'Adhesive Energy': {'results':{'GPTJ': 0.93,
                                        'Llama': 0.96,
                                        'Mistral': 0.89,
                                        'RF': 0.90,
                                        'XGBoost': 0.94},
                                    'epochs' : 4,
                                    'size': 5000
                                    },

                'Density\n(Monomers)':{'results':{'GPTJ': 0.88,
                                                'Llama': 0.87,
                                                'Mistral': 0.84,
                                                'RF': 0.76,
                                                'XGBoost': 0.75},
                                        'epochs': 20,
                                        'size': 300 
                                        },
                'Coh. E\n(Monomers)':{'results':{'GPTJ': 0.77,
                                                'Llama': 0.75,
                                                'Mistral': 0.78,
                                                'RF': 0.64,
                                                'XGBoost': 0.70},
                                        'epochs': 20,
                                        'size': 300 
                                        }, 
                'Sq. r\nof gyration\n(Monomers)':{'results':{'GPTJ': 0.91,
                                                'Llama': 0.91,
                                                'Mistral': 0.86,
                                                'RF': 0.89,
                                                'XGBoost': 0.88},
                                        'epochs': 20,
                                        'size': 300 
                                        },
                "$T_{g}$\n(Monomers)":{'results':{'GPTJ': 0.80,
                                                'Llama': 0.79,
                                                'Mistral': 0.84,
                                                'RF': 0.74,
                                                'XGBoost': 0.80},
                                        'epochs': 20,
                                        'size': 300 
                                        }, 
                'Melting Point\n(Molecules)':{'results':{'GPTJ': 0.686,
                                                'Llama': 0.586,
                                                'Mistral': 0.566,
                                                'RF': 0.68,
                                                'XGBoost': 0.66},
                                        'epochs': 20,
                                        'size': 1000 
                                        },
                'Dyn. Viscosity':{'results':{'GPTJ': 0.800,
                                                'Llama': 0.675,
                                                'Mistral': 0.642,
                                                'RF': 0.800,
                                                'XGBoost': 0.790},
                                        'epochs': 30,
                                        'size': 80 
                                        },    
                'Grain Size\n(Mg alloys)':{'results':{'GPTJ': 0.853,
                                                'Llama': 0.893,
                                                'Mistral': 0.940,
                                                'RF': 0.953,
                                                'XGBoost': 0.947},
                                        'epochs': 100,
                                        'size': 30 
                                        },      
                'LPS Propensity\n(Proteins)':{'results':{'GPTJ': 0.95,
                                                'Llama': 0.85,
                                                'Mistral': 0.92,
                                                'RF': 0.00,
                                                'XGBoost': 0.0,
                                                'In House': 0.895},
                                        'epochs': 25,
                                        'size': 75 
                                        },   
                'Structure\n(NPs)':{'results':{'GPTJ': 0.943,
                                                'Llama': 0.940,
                                                'Mistral': 0.967,
                                                'RF': 0.939,
                                                'XGBoost': 0.958},
                                        'epochs': 30,
                                        'size': 1800 
                                        },     
                'Melting Point\n(TAGs)':{'results':{'GPTJ': 0.92,
                                                'Llama': 0.92,
                                                'Mistral': 0.87,
                                                'RF': 0.873,
                                                'XGBoost': 0.886},
                                        'epochs': 25,
                                        'size': 150 
                                        }
}

reaction_results_dict = {'Click\nreaction': {'results':{'GPTJ': 0.940,
                                        'Llama': 0.846,
                                        'Mistral': 0.880,
                                        'RF': 0.893,
                                        'XGBoost': 0.913},
                                    'epochs' : 25,
                                    'size': 500
                                    },

                'Nickel\nCatalysis':{'results':{'GPTJ': 0.88,
                                                'Llama': 0.71,
                                                'Mistral': 0.79,
                                                'RF': 0.0,
                                                'XGBoost': 0.0},
                                        'epochs': 20,
                                        'size': 1000 
                                        },
                'Isomerization':{'results':{'GPTJ': 0.44,
                                                'Llama': 0.50,
                                                'Mistral': 0.50,
                                                'RF': 0.0,
                                                'XGBoost': 0.0},
                                        'epochs': 50,
                                        'size': 13 
                                        }, 
                'Kinetics\n(Polymers)':{'results':{'GPTJ': 0.7913,
                                                'Llama': 0.8348,
                                                'Mistral': 0.8087,
                                                'RF': 0.0,
                                                'XGBoost': 0.0},
                                        'epochs': 100,
                                        'size': 28 
                                        },
                '${H_{2}O}$ Splitting\n(HER) (MOFs)':{'results':{'GPTJ': 0.92,
                                                'Llama': 0.64,
                                                'Mistral': 0.89,
                                                'RF': 0.0,
                                                'XGBoost': 0.0},
                                        'epochs': 25,
                                        'size': 50 
                                        }, 
                '${CO_{2}}$ conversion\n(MOFs)':{'results':{'GPTJ': 0.68,
                                                'Llama': 0.60,
                                                'Mistral': 0.58,
                                                'RF': 0.60,
                                                'XGBoost': 0.57},
                                        'epochs': 100,
                                        'size': 65 
                                        }, 
                'MOF Synthesis':{'results':{'GPTJ': 1.0,
                                                'Llama': 0.80,
                                                'Mistral': 0.93,
                                                'RF': 0.80,
                                                'XGBoost': 1.0},
                                        'epochs': 50,
                                        'size': 15 
                                        },     
}


application_results_dict = {'He Diffusion\n(MOFs)': {'results':{'GPTJ': 0.72,
                                        'Llama': 0.70,
                                        'Mistral': 0.726,
                                        'RF': 0.766,
                                        'XGBoost': 0.726},
                                    'epochs' : 25,
                                    'size': 500
                                    },
                    #'CH4 Diffusion\n(MOFs)':{'results':{'GPTJ': 0.81,
                                                #'Llama': 0.0,
                                                #'Mistral': 0.0,
                                                #'RF': 0.8400,
                                               # 'XGBoost': 0.833},
                                        #'epochs': 0,
                                        #'size': 0 
                                        #},
                'H-Storage\n(Metal Hydrides)':{'results':{'GPTJ': 0.77,
                                                'Llama': 0.86,
                                                'Mistral': 0.79,
                                                'RF': 0.0,
                                                'XGBoost': 0.0},
                                        'epochs': 50,
                                        'size': 350 
                                        },
                '${CO_{2}}$ adsorption\n(Biomass)':{'results':{'GPTJ': 0.722,
                                                'Llama': 0.759,
                                                'Mistral': 0.722,
                                                'RF': 0.733,
                                                'XGBoost': 0.680},
                                        'epochs': 140,
                                        'size': 65 
                                        }, 
                'Thermal\nDesalination':{'results':{'GPTJ': 0.867,
                                                'Llama': 0.800,
                                                'Mistral': 0.867,
                                                'RF': 1.0,
                                                'XGBoost': 1.0},
                                        'epochs': 100,
                                        'size': 25 
                                        },
                'Detection Response\n(Gas Sensors)':{'results':{'GPTJ': 0.894,
                                                'Llama': 0.864,
                                                'Mistral': 0.879,
                                                'RF': 0.891,
                                                'XGBoost': 0.900},
                                        'epochs': 100,
                                        'size': 45 
                                        }, 
                'Stability\n(Gas Sensors)':{'results':{'GPTJ': 0.708,
                                                'Llama': 0.625,
                                                'Mistral': 0.625,
                                                'RF': 0.675,
                                                'XGBoost': 0.300},
                                        'epochs': 120,
                                        'size': 15 
                                        }, 
                'Gasification\n(Biomass)':{'results':{'GPTJ': 0.775,
                                                'Llama': 0.756,
                                                'Mistral': 0.680,
                                                'RF': 0.760,
                                                'XGBoost': 0.867},
                                        'epochs': 100,
                                        'size': 45 
                                        },     
}