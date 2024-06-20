import HardwareSetup as HS
from pydrake.all import (
    SceneGraphCollisionChecker,
    RandomGenerator,
    MosekSolver,
    GurobiSolver,
    SaveIrisRegionsYamlFile,
)
import pydrake.planning as plan

plant, diagram = HS.LoadRobotIRIS()

def get_regions():
    params = dict(edge_step_size=0.125)
    iiwa_model_instance_index = plant.GetModelInstanceByName("iiwa")
    wsg_model_instance_index = plant.GetModelInstanceByName("wsg")
    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = diagram
    checker = SceneGraphCollisionChecker(**params)

    options = plan.IrisFromCliqueCoverOptions()
    options.num_points_per_coverage_check = 5000
    options.num_points_per_visibility_round = 500
    options.coverage_termination_threshold = 0.7

    generator = RandomGenerator(0)

    if (MosekSolver().available() and MosekSolver().enabled()) or (
            GurobiSolver().available() and GurobiSolver().enabled()):
        # We need a MIP solver to be available to run this method.
        sets = plan.IrisInConfigurationSpaceFromCliqueCover(
            checker=checker, options=options, generator=generator,
            sets=[]
        )

        if len(sets) < 1:
            raise("No regions found")

        return sets
    else:
        print("No solvers available")

regions = get_regions()

SaveIrisRegionsYamlFile("CliqueCoverRegions.yaml", regions)