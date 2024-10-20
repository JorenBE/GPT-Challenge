# GPT-Challenge

Fine-tuned Large Language Models (LLMs) for Chemistry and Materials Science applications.


### Description

To fine-tuning our models we used the `chemlift` package, which can be installed from https://github.com/lamalab-org/chemlift.

The datasets, scripts, results, and examples for the studied case studies can be found in `experiments`.


### Datasets

The datasets can also be downloaded here.

| Dataset | Expert(s) | Affiliation | Data |
| -------- | -------- | -------- |-------- |
| Adhesive Energy of Polymers | Jiale Shi, Jonathan Whitmer | University of Notre Dame (USA) |[Download](experiments/01_Materials_and_Properties/AdE_polymers/DatasetExplore/train_polymers.csv?raw=true)|
| Properties of Monomers | KJ Schmidt, Ben Blaiszik, Ian T. Foster | University of Chicago (USA) | [Download](experiments/01_Materials_and_Properties/Prop_monomers/DatasetExplore/train_monomers.csv?raw=true) |
| Melting Point of Molecules | <p>Guillaume Godin</p><p>Igor Tetko</p> | <p>dsm-firmenich SA (Switzerland)</p><p>Helmholtz Munich; BIGCHEM GmbH (Germany)</p> | [Download](experiments/01_Materials_and_Properties/MeltingPoint_molecules/DatasetExplore/train_meltingPoint_noDuplicates.csv?raw=true) |
| Dynamic Viscosity of Molecules | Mahyar Rajabi-Kochic, Mohamad Moosavi | University of Toronto (Canada) | [Download](experiments/01_Materials_and_Properties/DynamicViscosity_molecules/Viscosity_dataset.csv?raw=true) |
| Microstructural Properties of Magnesium Alloys | Jianan Gu, Domonkos Tolnai, D.C. Florian Wieland, Regine Willumeit-Römer | Helmholtz Zentrum Hereon (Germany) | [Download](experiments/01_Materials_and_Properties/Prop_MgAlloys/DatasetExplore/HEREON_final.csv?raw=true) |
| Phase Separation Propensity of Proteins | Lydia Good, Alex Abrudan, Tuomas P.J. Knowles | University of Cambridge (UK) | [Download](experiments/01_Materials_and_Properties/PhaseSep_proteins/DatasetExplore/LLPS_all.csv?raw=true) |
| Structure of Nanoparticles | Andy S. Anker | Technical University of Denmark (Denmark) |[Download](experiments/01_Materials_and_Properties/Structure_nanoparticles/ScatteringPattern_dataset.csv?raw=true)|
| Melting Temperature of Triacylglycerols  | Antonio Buffo, Michele Lessona, Elena Simone | Politecnico di Torino (Italy) |[Download](experiments/01_Materials_and_Properties/MeltingPoint_TAGs/fats_learningCurve/fats_data.csv?raw=true)|
| Activation Energy of Cycloadditions | Dennis Svatunek | TU Wien (Austria) | [Download](experiments/02_Reactions_and_Synthesis/ActivationEnergy_cycloadditions/DatasetExplore/ClickActivationE.csv?raw=true) |
| Free Energy of Catalyzed Cleavage Reaction | Rubén Laplaza, Clemence Corminboeuf | École Polytechnique Fédérale de Lausanne (EPFL) (Switzerland) | [Download](experiments/02_Reactions_and_Synthesis/FreeEnergy_cleavageReact/DatasetExplore/NiCatalysis.csv?raw=true) |
| Yield of Catalytic Isomerization | Leander Choudhury | École Polytechnique Fédérale de Lausanne (EPFL) (Switzerland) | [Download](experiments/02_Reactions_and_Synthesis/Yield_isomerisation/DatasetExpore/Isomerisation_train.csv?raw=true) |
| Kinetics of Polymerization | Joren Van Herck, Tanja Junkers | Monash University (Australia) | [Download](experiments/02_Reactions_and_Synthesis/Kinetics_polymerization/DatasetExplore/Polymerization.csv?raw=true) |
| Photocatalytic Water Splitting Activity of MOFs | Beatriz Mouriño, Sauradeep Majumdar, Xin Jin | École Polytechnique Fédérale de Lausanne (EPFL) (Switzerland) | [Download](experiments/02_Reactions_and_Synthesis/Photocat_waterSplitting_MOFs/ExploreDataset/MOFs_photocatalysis.csv?raw=true) |
| Photocatalytic Carbondioxide Conversion Activity of MOFs | Matthew Garvin, Neda Poudineh, Susana Garcia, Ruaraidh D. McIntosh | Heriot-Watt University (UK) | [Download](experiments/02_Reactions_and_Synthesis/Photocat_CO2conversion_MOFs/PhotocatCO2conversionMOFs_dataset.csv?raw=true) |
| Success of MOF Synthesis | Francesca Nerli, Marco Taddei | Universitá di Pisa (Italy) | [Download](experiments/02_Reactions_and_Synthesis/MOF_synthesis/DatasetExplore/MOF_synthesis_train.csv?raw=true) |
| Gas Uptake and Diffusion of MOFs | Hilal Daglar, Seda Keskin | Koç University (Turkey) | [Download](experiments/03_Systems_and_Applications/GasUptakeDiffusion_MOFs/DatasetExplore/Helium.csv?raw=true) |
| Hydrogen Storage Capacity of Metal Hydrides | Noémie Xiao Hu, Andreas Züttel | École Polytechnique Fédérale de Lausanne (EPFL) (Switzerland) | [Download](experiments/03_Systems_and_Applications/H2storage_metalHydrides/DatasetExplore/Hydrides.csv?raw=true) |
| Carbondioxide Adsorption of Biomass-derived Adsorbents | Hossein Mashhadimoslem | University of Waterloo (Canada) | <p>[Download_BET](experiments/03_Systems_and_Applications/CO2ads_biomassAdsorbents/BET_dataset.csv?raw=true)</p><p>[Download_CO2ads](experiments/03_Systems_and_Applications/CO2ads_biomassAdsorbents/CO2adsorption_dataset.csv?raw=true)</p> |
| Thermal Desalination of Water | <p>Mehrdad Asgari</p><p>Morteza Sagharichiha</p> | <p>University of Cambridge (UK)</p><p>University of Tehran (Iran)</p> | [Download](experiments/03_Systems_and_Applications/Desalination_water/Desalination_dataset.csv?raw=true) |
| Detection Response of Gas Sensors | <p>Mehrdad Asgari</p><p>Fahimeh Hooriabad Saboor</p> | <p>University of Cambridge (UK)</p><p>University of Mohaghegh Ardabili (Iran)</p> | [Download](experiments/03_Systems_and_Applications/DetectionResponse_gasSensors/DetectionResponseSensors_dataset.csv?raw=true) |
| Stability of Gas Sensors | <p>Mehrdad Asgari</p><p>Sahar Vahdatifar</p> | <p>University of Cambridge (UK)</p><p>University of Tehran (Iran)</p> | [Download](experiments/03_Systems_and_Applications/Stability_gasSensors/StabilityGasSensors_dataset.csv?raw=true) |
| Gasification of Biomass | María Victoria Gil, Covadonga Pevida | Instituto de Ciencia y Tecnología del Carbono (INCAR), CSIC (Spain) | [Download](experiments/03_Systems_and_Applications/Gasification_biomass/GasificationBiomass_dataset.csv?raw=true) |
