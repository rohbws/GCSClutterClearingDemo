import numpy as np
from pydrake.all import Simulator
from pydrake.all import *
import queue
from pydrake.all import BasicVector
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.geometry.optimization import Point, GraphOfConvexSetsOptions
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LeafSystem,
    Parser,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    PiecewisePose,
    Quaternion,
)
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
import os.path
import time
from collections import OrderedDict
from typing import Dict

from pydrake.geometry import (
    StartMeshcat,
)
from pydrake.geometry.optimization import (
    IrisInConfigurationSpace,
    IrisOptions,
    LoadIrisRegionsYamlFile,
    SaveIrisRegionsYamlFile,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import PackageMap, Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant

from pydrake.multibody.tree import Body
from pydrake.solvers import Solve
from pydrake.systems.framework import DiagramBuilder, LeafSystem
import math

from manipulation.station import load_scenario
from pydrake.all import DiagramBuilder, Simulator

from iiwa_setup.iiwa import IiwaHardwareStationDiagram

from forward_kinematics import forward_kinematics, IiwaForwardKinematics

iris_filename = "my_iris.yaml"
iris_regions = dict()
q = []

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


# The trajOperator class is used to manage and operate trajectories for a robotic plant.
# It contains methods to get the current position of the robot, calculate the output position,
# and manage the state of the robot during the execution of a trajectory.

class trajOperator():
    # The constructor initializes the plant, trajectory type, trajectory, context, input port, and last position.
    # It also sets up some initial state variables.
    def __init__(self, plant, typeint, traj, context, iiwaInPort, lastP = [None]) -> None:
        self.plant = plant
        self.type = typeint
        self.traj = traj
        self.iiwaInPort = iiwaInPort
        self.gripper = self.plant.GetModelInstanceByName("wsg")
        self.firstCall = True
        self.startReached = False
        self.stackTime = context.get_time()
        self.lastPos = lastP

    # This method returns the current position of the robot.
    def getiiwaPos(self, context):
        return self.iiwaInPort.Eval(context)

    def getOutputPos(self, context):
        if not self.lastPos[0]:
            self.lastPos = self.getiiwaPos(context)

        if self.firstCall and self.startReached:
            plantCont = self.plant.CreateDefaultContext()
            
            self.plant.SetPositions(plantCont, self.lastPos)
            self.startEE = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()

            self.firstCall = False

        startDist = np.linalg.norm(self.getRunStartPos(context) - self.self.getiiwaPos(context))

        if startDist < 0.01:
            self.startReached = True

        if self.runToStartRequried() and not self.startReached:
            targPos = 0
            targPosCoeff = 0.9995
            targPosCoeff = math.atan(100.0*startDist) * 2.0 * 0.9995 * 0.3 / math.pi + 0.7
            targPos = targPosCoeff * np.array(self.lastPos) + (1-targPosCoeff) * self.getRunStartPos(context)

            targPos[0] = targPos[0] + 1.57

            return targPos

        if self.type == 0 or self.type == 4:
            newPos = np.array(self.traj.value(context.get_time() - self.stackTime)).flatten() * 0.2
        elif self.type == 1:
            plantCont = self.plant.CreateDefaultContext()
            curPos = self.lastPos.copy()
            curPos = np.concatenate((curPos, [0,0]))
            self.plant.SetPositions(plantCont, curPos)

            params = DifferentialInverseKinematicsParameters(self.plant.num_positions(),
                                                             self.plant.num_velocities())
            newPos = np.array(DoDifferentialInverseKinematics(self.plant, plantCont, [0, 0, 0, 0, 0,-1], self.plant.GetFrameByName("body", self.gripper), params).joint_velocities)[0:7]
            newPos = np.array(curPos)[0:7] + 0.001 * newPos
        elif self.type == 3:
            plantCont = self.plant.CreateDefaultContext()
            curPos = self.lastPos.copy()
            curPos = np.concatenate((curPos, [0,0]))
            self.plant.SetPositions(plantCont, curPos)

            params = DifferentialInverseKinematicsParameters(self.plant.num_positions(),
                                                             self.plant.num_velocities())
            newPos = np.array(DoDifferentialInverseKinematics(self.plant, plantCont, [0, 0, 0, 0, 0,1], self.plant.GetFrameByName("body", self.gripper), params).joint_velocities)[0:7]
            newPos = np.array(curPos)[0:7] + 0.001 * newPos
        else:
            newPos = self.lastPos

        
        self.lastPos = newPos.copy()
        newPos[0] = newPos[0] + 1.57
        return newPos

    def runToStartRequried(self) -> bool:
        return (self.type == 0 or self.type == 4)
    
    def getDistToStart(self, context) -> float:
        if self.runToStartRequried():
            return np.linalg.norm(self.getRunStartPos(context) - self.lastPos)

    def getRunStartPos(self, context):
        if self.runToStartRequried():
            if self.type == 0 or self.type == 4:
                return np.array(self.traje[self.currTraj].value(0)).flatten()
            else:
                raise TypeError("Run to start not required for this stage")
            
    def exitConditionReached(self, context, exitThresh = 0.01) -> bool:
        if self.type == 0 or self.type == 4:
            endPos = np.array(self.traj.value(self.traj.end_time())).flatten()
            return np.linalg.norm(endPos - self.lastPos) < exitThresh
        elif self.type == 1:
            plantCont = self.plant.CreateDefaultContext()
            self.plant.SetPositions(plantCont, self.lastPos)
            endPos = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()
            return (endPos[2] - self.traj[0]) > 0
        elif self.type == 3:
            plantCont = self.plant.CreateDefaultContext()
            self.plant.SetPositions(plantCont, self.lastPos)
            endPos = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()
            return (endPos[2] - self.traj[0]) < 0
        elif self.type == 2 or self.type == 5:
            return context.get_time() - self.stackTime - self.traj[0] > 0

class TrajPosOut(LeafSystem):

    def __init__(self, plant, plant_context):
        LeafSystem.__init__(self)
        self.traje = []
        self.plant = plant
        self.plant_context = plant_context

        self.iiwaPos = self.DeclareVectorInputPort("iiwa_measured", BasicVector(7))
        
        self.DeclareVectorOutputPort("iiwa_position", BasicVector(7), self.calciiwaPos)
        self.DeclareVectorOutputPort("plan_stage", BasicVector(1), self.getI)

        self.state = 0
        self.outTraj = 0
        self.stackTime = 0
        self.lastPos = [None]
        self.gcsPlan = GCSPlanner(self.plant)
        self.antGrasp = AntipodalGrasps()

        self.gcsPlan.addTrajEndPoints(seeds["Transition"], self.antGrasp.getNextGrasp()[0])
        self.planNextTraj()

        self.firstRun = True
        
                                     
    def getI(self, context, output):
        output.SetFromVector([self.state])

    def calcNextGCS(self):
        nextGrasp = self.antGrasp.getNextGrasp()
        graspUp = RigidTransform(nextGrasp.rotation(), nextGrasp.translation() + [0, 0, 0.1])
        self.gcsPlan.addTrajEndPoints(seeds["Deposit Pos 2"], graspUp)
        self.gcsPlan.addTrajEndPoints(graspUp, seeds["Deposit Pos 2"])
        self.planNextTraj()
        self.planNextTraj()


    def calciiwaPos(self, context, output):

        '''
        States:
        0 - running from bin to grasp up (gcs)
        1 - executing diff ik down
        2 - pausing to grasp
        3 - executing diff ik up
        4 - running to bin (gcs)
        5 - pausing to deposit


        Within each execution:
        if startPos is not None run to start position
        execute command
        - run until traj class determines execution conditions met
        - - determined purely based on context (time or position)
        repeat
        '''
        if self.firstRun:
            self.currTraj = trajOperator(self.plant, self.state, self.gcsPlan.getNextTraj(), context, self.iiwaPos, self.lastPos)
            self.firstRun = False

        if self.currTraj.exitConditionReached(context):
            if self.state == 2 or self.state == 5:
                self.calcNextGCS()
            
            self.state = (self.state + 1) % 6
            if self.state == 0 or self.state == 4:
                self.currTraj = trajOperator(self.plant, self.state, self.gcsPlan.getNextTraj(), context, self.iiwaPos, self.lastPos)
            elif self.state == 1 or self.state == 3:
                self.currTraj = trajOperator(self.plant, self.state, [0.1], context, self.iiwaPos, self.lastPos)
            else:
                self.currTraj = trajOperator(self.plant, self.state, [2], context, self.iiwaPos, self.lastPos)
            self.outTraj += 1
        
        output.SetFromVector(self.currTraj.getOutputPos(context))


class AntipodalGrasps():
    def __init__(self) -> None:
        self.graspPoses = []
    
    def getNextGrasp(self):
        return self.graspPoses[0]
    

class GCSPlanner():
    def __init__(self, plant):
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()
        self.trajs = queue.Queue()
        self.trajInps = queue.Queue()
        iris_filename = "my_iris.yaml"
        self.iris_regions = dict()
        self.iris_regions.update(LoadIrisRegionsYamlFile(iris_filename))

    def addTrajEndPoints(self, start, end):
        self.trajInps.put([start, end])

    def GcsTrajOpt(self, q_start, q_goal):
        if not self.iris_regions:
            print(
                "No IRIS regions loaded. Make some IRIS regions then come back and try this again."
            )
            return
        assert len(q_start) == len(q_goal)
        assert len(q_start) == self.iris_regions[next(iter(self.iris_regions))].ambient_dimension()

        edges = []

        gcs = GcsTrajectoryOptimization(len(q_start))
        regions = gcs.AddRegions(list(self.iris_regions.values()), order=1)
        source = gcs.AddRegions([Point(q_start)], order=0)
        target = gcs.AddRegions([Point(q_goal)], order=0)
        edges.append(gcs.AddEdges(source, regions))
        edges.append(gcs.AddEdges(regions, target))
        
        gcs.AddTimeCost()
        gcs.AddVelocityBounds(
            self.plant.GetVelocityLowerLimits(), (self.plant.GetVelocityUpperLimits())
        )

        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = 5
        start_time = time.time()
        print("Running GCS")
        traj, result = gcs.SolvePath(source, target, options)
        print(f"GCS solved in {time.time() - start_time} seconds")
        if not result.is_success():
            print("Could not find a feasible path from q_start to q_goal")
            return
        
        return traj

    def planNextTraj(self):
        nextTraj = self.trajInps.get()
        self.trajs.put(self.GcsTrajOpt(nextTraj[0], nextTraj[1]))

    def getNextTraj(self):
        return self.trajs.get()

class WSGOut(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("wsg_position", BasicVector(1), self.calciiwaPos)
        self.statusTime = self.DeclareVectorInputPort("statusTime", 1)
        self.closeTimes = [2, 3, 4, 8, 9, 10, 15, 16, 17, 22, 23, 24]
        
    def calciiwaPos(self, context, output):

        if self.statusTime.Eval(context)[0] in self.closeTimes:
            output.SetFromVector([0.0])
        else:
            output.SetFromVector([0.3])
        if self.statusTime.Eval(context)[0] > 1:
            output.SetFromVector([0.0])
        else:
            output.SetFromVector([0.3])
        
def make_environment_model_display(
    directive=None, draw=True, rng=None, num_ycb_objects=0, bin_name="bin0", has_wsg = True
) -> Body:
    # Make one model of the environment, but the robot only gets to see the sensor outputs.

    builder = DiagramBuilder()
    
    obj = None
    
    '''
    for i in range(num_ycb_objects):
        object_num = rng.integers(len(ycb))
        obj = parser.AddModelsFromUrl(
            f"package://manipulation/hydro/{ycb[object_num]}"
        )
    parser.package_map().AddRemote(
        package_name="gcs",
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/mpetersen94/gcs/archive/refs/tags/arxiv_paper_version.tar.gz"
            ],
            sha256=("6dd5e841c8228561b6d622f592359c36517cd3c3d5e1d3e04df74b2f5435680c"),
            strip_prefix="gcs-arxiv_paper_version",
        ),
    )
    '''
    scenario_data = (
        """
    directives:
    - add_directives:
        file: package://manipulation/iiwa_and_wsg.dmd.yaml
    plant_config:
        time_step: 0.005
        contact_model: "hydroelastic"
        discrete_contact_solver: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            hand_model_name: wsg
        wsg: !SchunkWsgDriver {}
    """
        if has_wsg
        else """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa7.dmd.yaml
    plant_config:
        # For some reason, this requires a small timestep
        time_step: 0.0001
        contact_model: "hydroelastic"
        discrete_contact_solver: "sap"
    model_drivers:
        iiwa: !IiwaDriver {}
    """
    )
    model_directives = """    
    directives:

    - add_directives:
        file: package://manipulation/clutter.dmd.yaml
        
    - add_model:
        name: foam_brick1
        file: package://manipulation/hydro/061_foam_brick.sdf
        default_free_body_pose:
            base_link:
                translation: [-0.17, -0.6, 0.05]
    - add_model:
        name: foam_brick2
        file: package://manipulation/hydro/061_foam_brick.sdf
        default_free_body_pose:
            base_link:
                translation: [-0.18, -0.5, 0.05]
            
        
    model_drivers:
        iiwa: !IiwaDriver
        hand_model_name: wsg
        wsg: !SchunkWsgDriver {}
    """

    scenario = load_scenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=True, use_hardware=True,
        ),
    )
    
    plant = station.get_internal_plant()
    
    scene_graph = station.GetSubsystemByName("external_station").GetSubsystemByName("scene_graph")
    
    robot = plant.GetModelInstanceByName("iiwa")
       
    AddRgbdSensors(builder, plant, scene_graph)
    
    iiwa_controller = builder.AddSystem(TrajPosOut(plant, station.get_internal_plant_context()))
    iiwa_controller.set_name("iiwa_controller")
    
    wsg_controller = builder.AddSystem(WSGOut(plant))
    wsg_controller.set_name("wsg_controller")
    
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), iiwa_controller.GetInputPort("iiwa_measured"))
    
    builder.Connect(wsg_controller.GetOutputPort("wsg_position"), station.GetInputPort("wsg.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("iiwa_position"), station.GetInputPort("iiwa.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("plan_stage"), wsg_controller.GetInputPort("statusTime"))

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
        
    return diagram, context, robot, plant, station.get_internal_plant_context()

# Another diagram for the objects the robot "knows about": gripper, cameras, bins.  Think of this as the model in the robot's head.
def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    plant.Finalize()
    return builder.Build()

# Sometimes it's useful to use inverse kinematics to find the seeds. You might
# need to adapt this to your robot. This helper takes an end-effector frame, E,
# and a desired pose for that frame in the world coordinates, X_WE.
def MyInverseKinematics(X_WE,plant, context, plantPos = [None]):
    # E = ee_body.body_frame()
    wsg = plant.GetModelInstanceByName("wsg")

    gripper_frame = plant.GetBodyByName("body", wsg).body_frame()

    ik = InverseKinematics(plant, context)

    ik.AddPositionConstraint(
        gripper_frame, [0, 0, 0], plant.world_frame(), X_WE.translation(), X_WE.translation(),
    )

    ik.AddOrientationConstraint(
        gripper_frame, RotationMatrix(), plant.world_frame(), X_WE.rotation(), 0.001,
    )
    
    prog = ik.get_mutable_prog()
    q = ik.q()
    q0 = plant.GetPositions(context)
    
    if not plantPos[0]:
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)
    else:
        prog.AddQuadraticErrorCost(np.identity(len(q)), plantPos, q)
        prog.SetInitialGuess(q, plantPos)
    
    result = Solve(ik.prog())
    if not result.is_success():
        print("IK failed")
        return plantPos

    return result.GetSolution(q)

def GenerateAntipodalGraspCandidate(
    diagram,
    context,
    cloud,
    rng,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        rng: a np.random.default_rng()
        wsg_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    index = rng.integers(0, cloud.size() - 1)

    # Use S for sample point/frame.
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)

    assert np.isclose(
        np.linalg.norm(n_WS), 1.0
    ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gx = n_WS  # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y, Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        return np.inf, None

    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0]

    # Try orientations from the center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = -R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        cost = GraspCandidateCost(diagram, context, cloud, adjust_X_G=True, verbose=False)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        if np.isfinite(cost):
            return cost, X_G

        # draw_grasp_candidate(X_G, f"collision/{theta:.1f}")

    return np.inf, None

# Note: The order of the seeds matters when we are using existing regions as
# configuration_obstacles.
seeds = OrderedDict()

seeds["ZeroedPosition"] = [0, 0, 0, 0, 0, 0, 0]
seeds["Above Bin 2"] = [0, 0.3, 0, -1.8, 0, 1, 1.57]
seeds["Deposit Pos 2"] = [-0.262, 0.3, 0, -1.8, 0, 1, (1.57)]
seeds["Deposit Pos 2 Up"] = [-0.262, 0.0, 0, -1.8, 0, 1, (1.57)]

seeds["In Bin 2"] = [0, 0.48, 0, -1.88, 0, 0.67, 1.57]
seeds["In Bin 1"] = [-1.56, 0.62, 0, -1.74, 0, 0.67, 1.57]
seeds["Above Bin 1"] = [-1.57, 0.3, 0, -1.8, 0, 1, 1.57]
seeds["Transition"] = [-0.75, -0.61, 0, -1.8, 0, 1, 1.57]

seeds["Between Bins"] = [-0.785, 0.2, 0, -1.8, 0, 1, 1.57]


def LoadRobot(plant: MultibodyPlant) -> Body:
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    # We'll use some tables, shelves, and bins from a remote resource.
    parser.package_map().AddRemote(
        package_name="gcs",
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/mpetersen94/gcs/archive/refs/tags/arxiv_paper_version.tar.gz"
            ],
            sha256=("6dd5e841c8228561b6d622f592359c36517cd3c3d5e1d3e04df74b2f5435680c"),
            strip_prefix="gcs-arxiv_paper_version",
        ),
    )

    
    model_directives = """    
directives:

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.3]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.8]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1]
        iiwa_joint_7: [1.57]

- add_weld:
    parent: world
    child: iiwa::base

# Add schunk
- add_model:
    name: wsg
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}

"""

    parser.AddModelsFromString(model_directives, ".dmd.yaml")

    extraGrip = plant.GetModelInstanceByName("gripper")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", extraGrip), xyz_rpy_deg([0, 0, 0], [0, 0, 0]))

    gripper = plant.GetModelInstanceByName("wsg")
    end_effector_body = plant.GetBodyByName("body", gripper)

meshcat2 = StartMeshcat()

    
def LoadRobotHardwareStation(builder = None):
    if not builder:
        builder = DiagramBuilder()
    
    model_directives = """    
directives:
- add_directives:
    file: package://manipulation/two_bins_w_cameras.dmd.yaml

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.3]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.8]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1]
        iiwa_joint_7: [1.57]

- add_weld:
    parent: world
    child: iiwa::base

# Add schunk
- add_model:
    name: wsg
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}
    """

    scenario = load_scenario(data=model_directives)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat = meshcat2))
    
    plant = station.GetSubsystemByName("plant")
    
    
    diagram = builder.Build()
    
    return plant, diagram

def LoadRobotHardwareStationHardware(builder = None):
    if not builder:
        builder = DiagramBuilder()
    
    model_directives = """    
directives:
- add_directives:
    file: package://manipulation/two_bins_w_cameras.dmd.yaml

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.3]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.8]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1]
        iiwa_joint_7: [1.57]

- add_weld:
    parent: world
    child: iiwa::base

# Add schunk
- add_model:
    name: wsg
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}
    """

    scenario = load_scenario(data=model_directives)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=True, use_hardware=False,
        ),
    )
    
    plant = station.get_internal_plant("plant")

    robotContext = station.get_internal_plant_context()
    
    
    diagram = builder.Build()
    
    return plant, diagram, robotContext

def convertToTraj(oldTraj, startPos = [None]):
    plantIK, diagramIK = LoadRobotHardwareStation()
    contextIK = diagramIK.CreateDefaultContext()
    plantContextiK = plantIK.GetMyContextFromRoot(contextIK)
    linMoveTrajPos = oldTraj.get_position_trajectory()
    linMoveTrajAng = oldTraj.get_orientation_trajectory()
    X_G = RigidTransform(Quaternion(linMoveTrajAng.value(0)), linMoveTrajPos.value(0))

    if not startPos[0]:
        startPos = MyInverseKinematics(X_G, plantIK, plantContextiK)
    
    trajReturn = []

    for t in np.append(
            np.arange(oldTraj.start_time(), oldTraj.end_time(), 0.01),
            oldTraj.end_time(),
        ):
        X_G = RigidTransform(Quaternion(linMoveTrajAng.value(t)), linMoveTrajPos.value(t))
        newPos = MyInverseKinematics(X_G, plantIK, plantContextiK, startPos)
        trajReturn.append(newPos)
        startPos = newPos

    return trajReturn

def move_schunk(trajs, dists):
    plantIK, diagramIk = LoadRobotHardwareStation()

    wsg = plantIK.GetModelInstanceByName("wsg")

    gripper_frame = plantIK.GetBodyByName("body", wsg).body_frame()

    contextIK = diagramIk.CreateDefaultContext()
    plantContextIK = plantIK.GetMyContextFromRoot(contextIK)

    oldPos = trajs[-1]

    try:
        oldPos = oldPos.value(oldPos.end_time())
    except:
        oldPos = oldPos[-1]

    plantIK.SetPositions(plantContextIK, oldPos)

    wsgPos = gripper_frame.CalcPoseInWorld(plantContextIK)

    wsgRot = wsgPos.rotation()

    wsgPosOnly = wsgPos.translation()

    endPos = RigidTransform(wsgRot, wsgPosOnly + dists)

    print(endPos)

    linMoveTraj = PiecewisePose.MakeLinear([0, (abs(np.linalg.norm(dists) * 10))], [wsgPos, endPos])

    linMoveTraj = convertToTraj(linMoveTraj, oldPos)


    return linMoveTraj

        
del iris_regions["GraspPos12"]
del iris_regions["GraspPos13"]
del iris_regions["GraspPos17"]
del iris_regions["GraspPos2"]
del iris_regions["GraspPos22"]
del iris_regions["GraspPos24"]


#The below regions have never been utilized in an antipodal grasp
del iris_regions["GraspPos10"]
del iris_regions["GraspPos18"]
del iris_regions["GraspPos19"]
del iris_regions["GraspPos20"]
del iris_regions["GraspPos21"]
del iris_regions["GraspPos3"]
del iris_regions["GraspPos4"]
del iris_regions["GraspPos5"]
#del iris_regions["GraspPos6"]
del iris_regions["GraspPos7"]
del iris_regions["GraspPos8"]
del iris_regions["GraspPos9"]
del iris_regions["GraspPos15"]
del iris_regions["Transition"]
del iris_regions["TransitionNoObs"]
del iris_regions["TransitionNoObs2"]

newCand1 = RigidTransform(
 R=RotationMatrix([
   [0.05565567547520678, -0.0002444713294406653, -0.9984499917477929],
   [0.998439517819967, -0.004573334192722878, 0.05565621142149002],
   [-0.00457985183498183, -0.9999895123690703, -1.0442167382385126e-05],
 ]),
 p=[-0.08398625420868182, -0.44822613274600115, 0.33908292428232244],
)


oldCand = RigidTransform(
 R=RotationMatrix([
   [0.05565567547520678, -0.0002444713294406653, -0.9984499917477929],
   [0.998439517819967, -0.004573334192722878, 0.05565621142149002],
   [-0.00457985183498183, -0.9999895123690703, -1.0442167382385126e-05],
 ]),
 p=[-0.08398625420868182, -0.44822613274600115, 0.23908292428232244],
)

jelloUpPos = [-2.28218173, 0.36183194, 0.6409329, -1.87001069, -0.25713113, 0.98730485, -1.4742302 ]

jelloUpPos[-1] += math.pi/2

jelloGrab = [-2.20756227, 0.53526387, 0.56340115, -1.94833742, -0.4047902, 0.76519159, -1.35562563]

jelloGrab[-1] += math.pi/2

tideUpPos = [-2.0033341, 0.57914925, 0.61913132, -1.56970203, -0.36290568, 1.10890311, -1.2293522 ]
tideUpPos[-1] += math.pi/2

tideGrab = [-1.95768116, 0.71346724, 0.55135146, -1.6421036, -0.44657148, 0.91555577, -1.16952908]
tideGrab[-1] += math.pi/2

cupGrab = [-2.15496806, 0.83227552, 0.46040382, -1.38088237, -0.3967169, 1.01938937, -1.62434109]
cupGrab[-1] += math.pi/8

cupUp = [-1.69340663, 0.68390589, -0.29562936, -1.3059921, 0.20039026, 1.17915618, -2.00987056]
cupUp[-1] += math.pi/8

glassUp = [-0.95816858, 0.58765876, -0.52798669, -1.61296202, 0.34795992, 1.00571432, (3.05432619 - (math.pi / 7))]
glassUpBigger = [-0.95816858, 0.30765876, -0.52798669, -1.61296202, 0.34795992, 1.00571432, (3.05432619 - (math.pi / 7))]

glassGrab = [-0.95816858, 0.69765876, -0.52798669, -1.61296202, 0.34795992, 1.00571432, (3.05432619 - (math.pi / 7))]

Rstraight=RotationMatrix([
    [0.05597484092032722, -0.0010961886872575403, -0.998431577803059],
    [0.9984194494229444, -0.004988304216565806, 0.055979637682178125],
    [-0.005041844695051499, -0.9999869575106491, 0.0008152365706108688],
  ])

diagram, context, robot, objPos, plant, internalPlantContext = make_environment_model_display(
    rng=np.random.default_rng(), num_ycb_objects=1, draw=True
)
wsg = plant.GetModelInstanceByName("wsg")

simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)
simulator.AdvanceTo(600.0)
