import numpy as np
from pydrake.all import Simulator
from pydrake.all import *
from IPython.display import clear_output
from pydrake.all import BasicVector
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RandomGenerator,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
    PiecewisePose,
    JacobianWrtVariable,
    Quaternion,
)
from manipulation.station import MakeHardwareStation, load_scenario, LoadScenario
from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
import multiprocessing as mp
import os.path
import time
from collections import OrderedDict
from typing import Dict

import numpy as np
import pydot
from IPython.display import SVG, display, display_jpeg, display_pdf, display_png, Image
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    MeshcatVisualizer,
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
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import PackageMap, Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant

from pydrake.multibody.tree import Body
from pydrake.solvers import GurobiSolver, MathematicalProgram, Solve
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer
import math

from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import load_scenario
from pydrake.all import DiagramBuilder, MeshcatVisualizer, Simulator

from iiwa_setup.iiwa import IiwaHardwareStationDiagram

from forward_kinematics import forward_kinematics, IiwaForwardKinematics

import pickle


iris_filename = "my_iris.yaml"
iris_regions = dict()
q = []

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

class TrajPosOut(LeafSystem):
    def __init__(self, plant, traj, plant_context):
        LeafSystem.__init__(self)
        self.traje = np.array(traj)
        self.plant = plant
        self.plant_context = plant_context
        self.gripper = self.plant.GetModelInstanceByName("wsg")

        #self.plantIK, self.contextIK = LoadRobotHardwareStation()

        self.iiwaPos = self.DeclareVectorInputPort("iiwa_measured", BasicVector(7))
        
        self.DeclareVectorOutputPort("iiwa_position", BasicVector(7), self.calciiwaPos)
        self.DeclareVectorOutputPort("plan_stage", BasicVector(1), self.getI)
        self.currTraj = 0
        self.stackTime = 0
        self.outTraj = 0
        self.prevTime = 0
        self.first = True
        self.prevout = [None]
        '''
        trajectory following key:
        negative - pause for the abs of that number of seconds
        0 - follow gcs traj
        1 - follow EE traj
        2 - follow list of poses
        3 - move with diff ik
        '''
        self.order = [0, 3, -2, 0, 0, -2, 0, 3, -2, 0, 0, -2, 0, 0, 3, -2, 0, 0, -2, 0, 0, 3,2, -2, 0,0, -2]
        self.endCondTime = False
        self.startTime = 0
        self.firstRunToStart = True
        self.lastPos = [None]
        self.time = 0

        self.fastMoves = [3, 4, 9, 10, 15, 16]
                                     
    def getI(self, context, output):
        output.SetFromVector([self.outTraj])


    def calciiwaPos(self, context, output):

        #print(self.outTraj)
        #print(self.currTraj)
        endPos = self.calcEndPose(context)

        distToEnd = self.calcDistToEnd(context, endPos)

        #print(distToEnd)

        if self.incrTrajVal(context, distToEnd):
            self.stackTime = context.get_time()

        startPos = self.calcStartPos(context)

        dist = np.linalg.norm(startPos - np.array(self.iiwaPos.Eval(context)))

        if (self.lastPos[0]):
            dist = np.linalg.norm(self.lastPos - startPos)

        if dist > 0.009 and self.first and self.order[self.outTraj] >= 0:
            #print(dist)
            posVal = np.array(self.runToStart(context, startPos, dist)).flatten()
            output.SetFromVector(posVal)
            self.lastPos = posVal
        elif self.first:
            self.stackTime = context.get_time()
            self.first = False
        else:
            posVal = np.array(self.determinePoseVal(context)).flatten()
            output.SetFromVector(posVal)
            self.lastPos = posVal

    def calcEndPose(self, context):
        endPos = [0]
        
        endPos = self.traje[self.currTraj][-1]

        endPos = endPos.copy()
        
        endPos[0] = endPos[0] + 1.57
        return endPos
    
    def calcStartPos(self, context):
        startPos = [0]
        
        startPos = self.traje[self.currTraj][0]

        startPos = startPos.copy()
        
        startPos[0] = startPos[0] + 1.57
        return startPos


    def calcDistToEnd(self, context, endPos):
        
        return np.linalg.norm(endPos - np.array(self.iiwaPos.Eval(context)))

    def incrTrajVal(self, context, distToEnd):
            if (distToEnd < 0.01) or self.time >=1.0:
                self.time = 0
                self.first = True
                if (self.outTraj < (len(self.order) - 1)):
                    self.currTraj += 1
                self.outTraj += 1
                self.loadNewEndCondTime()
                return True
        
            return False
    
    def loadNewEndCondTime(self):
        if self.order[self.outTraj] < 0:
            self.endCondTime == abs(self.order[self.outTraj])
        else:
            self.endCondTime = 0

    def runToStart(self, context, startPos, dist):
        targPos = 0
        targPosCoeff = 0.9995
        targPosCoeff = math.atan(100.0*dist) * 2.0 * 0.9995 * 0.3 / math.pi + 0.7
        if not self.lastPos[0]:
            self.prevout = self.iiwaPos.Eval(context)
            self.firstRunToStart = False
            targPos = targPosCoeff * np.array(self.prevout) + (1-targPosCoeff) * np.array(startPos)
        else:
            targPos = targPosCoeff * np.array(self.lastPos) + (1-targPosCoeff) * np.array(startPos)
        self.prevout = targPos
        return targPos
        
    
    def determinePoseVal(self, context):
        
        newPos = [None]
        currTime = context.get_time()

        t = ((currTime - self.stackTime) * 0.25)
        t = min(t, 1.0)

        self.time = t
        
        newPos = ((1-t)**3) * self.traje[self.currTraj][0] + 3 * t * ((1-t)**2) * self.traje[self.currTraj][1] + 3 * (t**2) * (1-t) * self.traje[self.currTraj][2] + (t**3) * self.traje[self.currTraj][3]
        
        newPos[0] = newPos[0] + 1.57

        return newPos


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
    traj, directive=None, draw=True, rng=None, num_ycb_objects=0, bin_name="bin0", has_wsg = True
) -> Body:
    # Make one model of the environment, but the robot only gets to see the sensor outputs.

    builder = DiagramBuilder()
    
    obj = None

    scenario_data = (
        """
    directives:
    - add_directives:
        file: package://manipulation/iiwa_and_wsg.dmd.yaml
    plant_config:
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            hand_model_name: wsg
            control_mode: position_only
        wsg: !SchunkWsgDriver {}
    lcm_buses:
        default:
            lcm_url: ""
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
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            control_mode: position_only
    lcm_buses:
        default:
            lcm_url: ""
    """
    )

    builder = DiagramBuilder()

    scenario = LoadScenario(data=scenario_data)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=has_wsg, use_hardware=True
        ),
    )

    
    plant = station.get_internal_plant()
    
    scene_graph = station.GetSubsystemByName("external_station").GetSubsystemByName("scene_graph")
    
    robot = plant.GetModelInstanceByName("iiwa")

    #gripper = plant.GetModelInstanceByName("gripper")
    #end_effector_body = plant.GetBodyByName("body", gripper)
       
    AddRgbdSensors(builder, plant, scene_graph)
    
    iiwa_controller = builder.AddSystem(TrajPosOut(plant, traj, station.get_internal_plant_context()))
    iiwa_controller.set_name("iiwa_controller")
    
    wsg_controller = builder.AddSystem(WSGOut(plant))
    wsg_controller.set_name("wsg_controller")
    
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), iiwa_controller.GetInputPort("iiwa_measured"))
    
    builder.Connect(wsg_controller.GetOutputPort("wsg_position"), station.GetInputPort("wsg.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("iiwa_position"), station.GetInputPort("iiwa.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("plan_stage"), wsg_controller.GetInputPort("statusTime"))

    diagram = builder.Build()
    #RenderDiagram(diagram)
    
    #simulator = Simulator(diagram)
    #simulator.set_target_realtime_rate(0.1)
    #simulator.AdvanceTo(2.0)
    context = diagram.CreateDefaultContext()

    objPos = [0]

        
    return diagram, context, robot, objPos, plant, station.get_internal_plant_context()

def make_environment_model(
    directive=None, draw=False, rng=None, num_ycb_objects=0, bin_name="bin0", plant=None, scene_graph=None, builder=None
) -> Body:
    # Make one model of the environment, but the robot only gets to see the sensor outputs.
    if not directive:
        directive = "package://manipulation/clutter_planning.dmd.yaml"

    #builder = DiagramBuilder()
    #plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.SetAutoRenaming(True)
    parser.AddModelsFromUrl(directive)
    obj = []

    for i in range(num_ycb_objects):
        object_num = rng.integers(len(ycb))
        continue
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
    
    can_directives = """
    
directives:

- add_model:
     name: pickObject
     file: package://manipulation/hydro/010_potted_meat_can.sdf
     default_free_body_pose:
         base_link_meat:
             translation: [-0.05, -0.45, 0.05]
            """
    
    brick_directives = """
directives:

- add_model:
    name: foam_brick
    file: package://manipulation/hydro/061_foam_brick.sdf
    default_free_body_pose:
        base_link:
            translation: [-0.17, -0.6, 0.05]
- add_model:
    name: foam_brick
    file: package://manipulation/hydro/061_foam_brick.sdf
    default_free_body_pose:
        base_link:
            translation: [-0.18, -0.5, 0.05]

            """
    sugar_directives = """
directives:

- add_model:
     name: pickObject
     file: package://manipulation/hydro/004_sugar_box.sdf
     default_free_body_pose:
         base_link_sugar:
             translation: [0.1, -0.55, 0.05]
             rotation: !Rpy { deg: [0.0, 90.0, 0.0 ]}
    """
    
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
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 90.0 ]}

"""

    robot = parser.AddModelsFromString(model_directives, ".dmd.yaml")
    obj.append(parser.AddModelsFromString(sugar_directives, ".dmd.yaml"))
    #obj.append(parser.AddModelsFromString(brick_directives, ".dmd.yaml"))
    #obj.append(parser.AddModelsFromString(can_directives, ".dmd.yaml"))

    extraGrip = plant.GetModelInstanceByName("gripper")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", extraGrip), xyz_rpy_deg([0, 0, 0], [0, 0, 0]))

    gripper = plant.GetModelInstanceByName("wsg")
    end_effector_body = plant.GetBodyByName("body", gripper)
    
    plant.Finalize()
    AddRgbdSensors(builder, plant, scene_graph)

    if draw:
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(prefix="environment"),
        )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    objPos = None
    
    sim = Simulator(diagram, context)
    sim.AdvanceTo(2.0)
    
    return diagram, context, robot, objPos, obj[0]

    if num_ycb_objects > 0:
        generator = RandomGenerator(rng.integers(1000))  # this is for c++
        plant_context = plant.GetMyContextFromRoot(context)
        bin_instance = plant.GetModelInstanceByName(bin_name)
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
        z = 0.2
        for body_index in plant.GetFloatingBaseBodies():
            tf = RigidTransform(
                UniformlyRandomRotationMatrix(generator),
                [rng.uniform(-0.15, 0.15), rng.uniform(-0.2, 0.2), z],
            )
            '''
            plant.SetFreeBodyPose(
                plant_context, plant.get_body(body_index), X_B.multiply(tf)
            )
            z += 0.1
            '''

        simulator = Simulator(diagram, context)
        simulator.AdvanceTo(2.0 if running_as_notebook else 0.1)
        objPos = plant.GetPositions(plant_context, obj[0])
    elif draw:
        diagram.ForcedPublish(context)
    diagram.ForcedPublish(context)
    return diagram, context, robot, objPos, obj[0]

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



def draw_grasp_candidate(X_G, prefix="gripper", draw_frames=True):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl(
        "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
    )
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
    plant.Finalize()

    # frames_to_draw = {"gripper": {"body"}} if draw_frames else {}
    params = MeshcatVisualizerParams()
    params.prefix = prefix
    params.delete_prefix_on_initialization_event = False
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)


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

if os.path.isfile(iris_filename):
    iris_regions.update(LoadIrisRegionsYamlFile(iris_filename))
    print(f"Loaded iris regions from {iris_filename}.")


from pydrake.planning import GcsTrajectoryOptimization
from pydrake.geometry.optimization import Point, GraphOfConvexSetsOptions


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



def GcsTrajOpt(q_start, q_goal):
    if not iris_regions:
        print(
            "No IRIS regions loaded. Make some IRIS regions then come back and try this again."
        )
        return
    assert len(q_start) == len(q_goal)
    assert len(q_start) == iris_regions[next(iter(iris_regions))].ambient_dimension()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    LoadRobot(plant)
    plant.Finalize()
    #AddDefaultVisualization(builder, meshcat)
    #diagram = builder.Build()
    edges = []

    gcs = GcsTrajectoryOptimization(len(q_start))
    # TODO(russt): AddRegions should take named regions.
    regions = gcs.AddRegions(list(iris_regions.values()), order=1)
    source = gcs.AddRegions([Point(q_start)], order=0)
    target = gcs.AddRegions([Point(q_goal)], order=0)
    edges.append(gcs.AddEdges(source, regions))
    edges.append(gcs.AddEdges(regions, target))
    
    gcs.AddTimeCost()
    gcs.AddVelocityBounds(
        plant.GetVelocityLowerLimits(), (plant.GetVelocityUpperLimits())
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
    
    for edge in gcs.graph_of_convex_sets().Edges():
        print(result.GetSolution(edge.phi()))

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

def move_down_finalPose(trajs, dist):
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

    endPos = RigidTransform(wsgRot, wsgPosOnly - [0, 0, dist])

    return endPos.translation()

def move_down(trajs, dist):
    plantIK, diagramIk = LoadRobotHardwareStation()

    wsg = plantIK.GetModelInstanceByName("wsg")

    gripper_frame = plantIK.GetBodyByName("body", wsg).body_frame()

    contextIK = diagramIk.CreateDefaultContext()
    plantContextIK = plantIK.GetMyContextFromRoot(contextIK)

    oldPos = trajs[-2]

    try:
        oldPos = oldPos.value(oldPos.end_time())
    except:
        oldPos = oldPos[-1]

    plantIK.SetPositions(plantContextIK, oldPos)

    wsgPos = gripper_frame.CalcPoseInWorld(plantContextIK)

    wsgRot = wsgPos.rotation()

    wsgPosOnly = wsgPos.translation()

    endPos = RigidTransform(wsgRot, wsgPosOnly - [0, 0, dist])

    linMoveTraj = PiecewisePose.MakeLinear([0, (abs(dist * 10))], [wsgPos, endPos])

    linMoveTraj = convertToTraj(linMoveTraj, oldPos)

    return linMoveTraj

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

def move_schunkAng(trajs, dists, angs):
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

    endPos = RigidTransform(RollPitchYaw(angs), wsgPosOnly + dists)

    print(endPos)

    linMoveTraj = PiecewisePose.MakeLinear([0, (abs(np.linalg.norm(dists) * 10))], [wsgPos, endPos])

    linMoveTraj = convertToTraj(linMoveTraj, oldPos)


    return linMoveTraj

def move_schunkAngRotMat(trajs, dists, angs):
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

    endPos = RigidTransform(angs, wsgPosOnly + dists)

    print(endPos)

    linMoveTraj = PiecewisePose.MakeLinear([0, (abs(np.linalg.norm(dists) * 10))], [wsgPos, endPos])

    linMoveTraj = convertToTraj(linMoveTraj, oldPos)


    return linMoveTraj



if (False):
    with open('trajs.pickle', 'rb') as f:
        trajs = pickle.load(f)
else:
    trajs = []

assert (
    seeds
), "The examples here use the 'manually-specified seeds' from the  section above. Please run that section first, or populate your own start and end configurations."


jelloUpPos = [-2.28218173, 0.36183194, 0.6409329, -1.87001069, -0.25713113, 0.98730485, -1.4742302 ]

jelloUpPos[-1] += math.pi/2

#trajs.append(GcsTrajOpt(seeds["Transition"], jelloUpPos))

import yaml

def extract(file_name:str):
    with open(file_name, "r") as yaml_file:
        extracted_list = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return extracted_list

policy_trajs = extract("traj2.yaml")

for traj in policy_trajs:
    trajs.append(traj)

diagram, context, robot, objPos, plant, internalPlantContext = make_environment_model_display(
    trajs, rng=np.random.default_rng(), num_ycb_objects=1, draw=True
)

simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)
simulator.AdvanceTo(600.0)
