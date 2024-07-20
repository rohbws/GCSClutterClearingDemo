import numpy as np
from pydrake.all import *
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    RigidTransform,
    RotationMatrix,
)
from manipulation.station import MakeHardwareStation
from manipulation.scenarios import AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

from pydrake.multibody.tree import Body
from pydrake.solvers import Solve
from pydrake.systems.framework import DiagramBuilder

from manipulation.station import load_scenario, LoadScenario, AddPointClouds, AppendDirectives

from manipulation import FindResource

import warnings

from pydrake.all import DiagramBuilder
from manipulation.systems import ExtractPose
from scipy.spatial.transform import Rotation as R

try:
    from iiwa_setup.iiwa import IiwaHardwareStationDiagram
except:
    warnings.warn("Could not import hardware functions", UserWarning)
    


def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def make_environment_model_hardware(
    hardware = False, has_wsg = True, traj = None, TrajPosOut = None, WSGOut = None,GraspSelector = None, AntipHandler = None, CameraPoseInWorldSource = None,
) -> Body:
    # Make one model of the environment, but the robot only gets to see the sensor outputs.

    builder = DiagramBuilder()
    
    obj = None

    scenario_data = (
        """
    directives:
    - add_directives:
        file: package://manipulation/iiwa_and_wsg.dmd.yaml
    - add_frame:
        name: camera0_staging
        X_PF:
            base_frame: world
            rotation: !Rpy { deg: [0, 0, 0.0]}
    - add_frame:
        name: handeye_camera_origin
        X_PF:
            base_frame: iiwa::iiwa_link_7
            rotation: !Rpy { deg: [2.1243414 ,  -0.45993413, -89.88104678]}
            translation: [-0.0697802, 0.0297534, 0.159606]
    - add_frame:
        name: camera0_origin
        X_PF:
            base_frame: camera0_staging
            rotation: !Rpy { deg: [-110, 0, 90]}
            translation: [1.2, 0.0, 0.54]
    - add_model:
        name: camera_main
        file: package://manipulation/camera_box.sdf
    - add_weld:
        parent: camera0_origin
        child: camera_main::base
    cameras:
        main_camera:
            name: camera0
            lcm_bus: driver_traffic
            depth: True
            X_PB:
                base_frame: camera_main::base
        handeye_camera:
            name: handeye_camera
            lcm_bus: driver_traffic
            depth: True
            X_PB:
                base_frame: handeye_camera_origin
    plant_config:
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    lcm_buses:
        driver_traffic:
            # Use a non-default LCM url to communicate with the robot.
            lcm_url: udpm://239.255.76.67:7667?ttl=0
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: driver_traffic
            hand_model_name: wsg
            control_mode: position_only
        wsg: !SchunkWsgDriver
            lcm_bus: driver_traffic

    camera_ids:
        handeye_camera: DRAKE_RGBD_CAMERA_IMAGES_810512062206
    """
    )

    builder = DiagramBuilder()

    scenario = LoadScenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, hardware=False))

    external_station = builder.AddSystem(MakeHardwareStation(scenario, hardware=True))

    handeye_camera_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 639.036, 639.036, 425.131, 244.165)))

    builder.Connect(external_station.GetOutputPort("handeye_camera.depth_image"), handeye_camera_pcd.GetInputPort("depth_image"))
    
    plant = station.GetSubsystemByName("plant")
        
    robot = plant.GetModelInstanceByName("iiwa")
    
    r = R.from_quat([0.0102867, -0.0159347, -0.706193, 0.707766])

    x_ee_camera = RigidTransform(
        R=RotationMatrix(r.as_matrix()),
        p = [-0.0697802, 0.0297534, 0.159606]
    )    
    camera_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_ee_camera))
    
    eef_pose = builder.AddSystem(
        ExtractPose(
            plant.GetBodyByName("iiwa_link_7").index()
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        eef_pose.get_input_port(),
    )
    builder.Connect(
        eef_pose.get_output_port(),
        camera_pose_source.GetInputPort("X_EE")
    )    
    
    builder.Connect(
        camera_pose_source.GetOutputPort("X_WC"),
        handeye_camera_pcd.GetInputPort("camera_pose"),
    )

    y_bin_grasp_selector = builder.AddSystem(
        GraspSelector(
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera_main"))[0],
            ],
        )
    )

    builder.Connect(
        handeye_camera_pcd.GetOutputPort("point_cloud"),
        y_bin_grasp_selector.GetInputPort("cloud0_W"),
    )

    builder.Connect(
        station.GetOutputPort("body_poses"),
        y_bin_grasp_selector.GetInputPort("body_poses"),
    )
    AntipHandler = builder.AddSystem(AntipHandler())

    
    iiwa_controller = builder.AddSystem(TrajPosOut(plant, plant.CreateDefaultContext()))
    iiwa_controller.set_name("iiwa_controller")
    
    wsg_controller = builder.AddSystem(WSGOut(plant))
    wsg_controller.set_name("wsg_controller")
    
    builder.Connect(y_bin_grasp_selector.GetOutputPort("grasp_selection"), AntipHandler.GetInputPort("grasps"))

    builder.Connect(AntipHandler.GetOutputPort("grasp_position"), iiwa_controller.GetInputPort("grasps"))

    builder.Connect(AntipHandler.GetOutputPort("grasp_cost"), iiwa_controller.GetInputPort("grasp_cost"))
    
    builder.Connect(external_station.GetOutputPort("iiwa.position_measured"), iiwa_controller.GetInputPort("iiwa_measured"))

    builder.Connect(external_station.GetOutputPort("iiwa.position_measured"), station.GetInputPort("iiwa.position"),)
    
    wsg_state_demux: Demultiplexer = builder.AddSystem(Demultiplexer(2, 1))
    
    builder.Connect(
            external_station.GetOutputPort("wsg.state_measured"),
            wsg_state_demux.get_input_port(),
    )
    
    builder.Connect(
            wsg_state_demux.get_output_port(0),
            station.GetInputPort("wsg.position"),
    )
    
    builder.Connect(wsg_controller.GetOutputPort("wsg_position"), external_station.GetInputPort("wsg.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("iiwa_position"), external_station.GetInputPort("iiwa.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("plan_stage"), wsg_controller.GetInputPort("statusTime"))

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
        
    return diagram, context, robot, plant, plant.CreateDefaultContext()


# Sometimes it's useful to use inverse kinematics to find the seeds. You might
# need to adapt this to your robot. This helper takes an end-effector frame, E,
# and a desired pose for that frame in the world coordinates, X_WE.
def MyInverseKinematics(X_WE,plant, context, plantPos = [-1.57, 0.3, 0, -1.8, 0, 1, 1.57]):
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
    
    if plantPos[0]:
        prog.AddQuadraticErrorCost(np.identity(len(q)), plantPos, q)
        prog.SetInitialGuess(q, plantPos)
    
    result = Solve(ik.prog())
    if not result.is_success():
        print("IK failed")
        return plantPos

    return result.GetSolution(q)

def LoadRobotIRISSeeds():
    builder = DiagramBuilder()
    model_directives = (
        """    
directives:

- add_directives:
        file: package://manipulation/two_bins_w_cameras.dmd.yaml

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
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
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}
    """)

    scenario = LoadScenario(data=model_directives)
    station = builder.AddSystem(MakeHardwareStation(scenario))
    
    plant = station.GetSubsystemByName("plant")
    
    diagram = builder.Build()
    
    return plant, diagram

def LoadRobotIRIS(builder = None):

    import os

    binPath = os.path.abspath("bin.sdf")
    uri = "file://{}".format(binPath)
    print(uri)
    print(binPath)
    if not builder:
        builder = RobotDiagramBuilder()
    
    model_directives = (
        """    
directives:

- add_model:
    name: bin0
    file: file:///home/rohanbosworth/manipulation/manipulation/models/hydro/bin.sdf

- add_weld:
    parent: world
    child: bin0::bin_base
    X_PC:
      rotation: !Rpy { deg: [0.0, 0.0, 90.0 ]}
      translation: [-0.05, -0.5, -0.015]

- add_model:
    name: bin1
    file: file:///home/rohanbosworth/manipulation/manipulation/models/hydro/bin.sdf

- add_weld:
    parent: world
    child: bin1::bin_base
    X_PC:
      rotation: !Rpy { deg: [0.0, 0.0, 180.0 ]}
      translation: [0.5, 0.05, -0.015]

- add_model:
    name: floor
    file: file:///home/rohanbosworth/manipulation/manipulation/models/floor.sdf

- add_weld:
    parent: world
    child: floor::box
    X_PC:
        translation: [0, 0, -.5]

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
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
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}
    """)

    model_directivesTwo = (
        """   
directives:
- add_model:
    name: bin0
    file: file:///home/rohanbosworth/manipulation/manipulation/models/hydro/bin.sdf

- add_weld:
    parent: world
    child: bin0::bin_base
    X_PC:
      rotation: !Rpy { deg: [0.0, 0.0, 90.0 ]}
      translation: [-0.05, -0.5, -0.015]

- add_model:
    name: bin1
    file: file:///home/rohanbosworth/manipulation/manipulation/models/hydro/bin.sdf

- add_weld:
    parent: world
    child: bin1::bin_base
    X_PC:
      rotation: !Rpy { deg: [0.0, 0.0, 180.0 ]}
      translation: [0.5, 0.05, -0.015]

- add_model:
    name: floor
    file: file:///home/rohanbosworth/manipulation/manipulation/models/floor.sdf

- add_weld:
    parent: world
    child: floor::box
    X_PC:
        translation: [0, 0, -.5]

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
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

    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

 

- add_weld:

    parent: iiwa::iiwa_link_7

    child: wsg::body

    X_PC:

      translation: [0, 0, 0.114]

      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}

 

# Add iiwa2

- add_model:

    name: iiwa2

    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf

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

    child: iiwa2::base

    X_PC:

        translation: [0.5, -0.5, 0.0]

        rotation: !Rpy { deg: [0.0, 0.0, 0.0 ]}

 

# Add schunk

- add_model:

    name: wsg2

    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

 

- add_weld:

    parent: iiwa2::iiwa_link_7

    child: wsg2::body

    X_PC:

      translation: [0, 0, 0.114]

      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}

    """)

    scenario = LoadScenario(data=model_directives)
    robot_model_instances = builder.parser().AddModelsFromString(model_directives, ".dmd.yaml")
    
    plant = builder.plant()
    
    diagram = builder.Build()
    
    return plant, diagram

def LoadRobotHardwareStation(builder = None):
    if not builder:
        builder = DiagramBuilder()
    
    model_directives = (
        """    
directives:
- add_directives:
    file: package://manipulation/two_bins.dmd.yaml

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
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
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}
    """)

    scenario = LoadScenario(data=model_directives)
    station = builder.AddSystem(MakeHardwareStation(scenario))
    
    plant = station.GetSubsystemByName("plant")
    
    diagram = builder.Build()
    
    return plant, diagram

rng = np.random.default_rng(135)  # this is for python


# Another diagram for the objects the robot "knows about": gripper, cameras, bins.  Think of this as the model in the robot's head.
def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    plant.Finalize()
    return builder.Build()

def make_environment_model_sim(
     TrajPosOut = None, WSGOut = None, GraspSelector = None, AntipHandler = None
) -> Body:
    # Make one model of the environment, but the robot only gets to see the sensor outputs.

    builder = DiagramBuilder()

    mesh = StartMeshcat()
    
    obj = None

    OLDscenario_data = (
        """
    directives:
    - add_directives:
        file: package://manipulation/two_bins_w_cameras.dmd.yaml
    
    - add_model:
        name: foam_brick1
        file: package://manipulation/hydro/061_foam_brick.sdf
        default_free_body_pose:
            base_link:
                translation: [0.6, -0.17, 0.05]
    - add_model:
        name: foam_brick2
        file: package://manipulation/hydro/061_foam_brick.sdf
        default_free_body_pose:
            base_link:
                translation: [0.5, -0.18, 0.05]
                
    - add_model:
        name: meatCan
        file: package://manipulation/hydro/010_potted_meat_can.sdf
        default_free_body_pose:
            base_link_meat:
                translation: [0.45, -0.05, 0.05]
                
    - add_model:
        name: pickObject
        file: package://manipulation/hydro/004_sugar_box.sdf
        default_free_body_pose:
            base_link_sugar:
                translation: [0.55, 0.05, 0.05]
                rotation: !Rpy { deg: [0.0, 90.0, 0.0 ]}
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
    )

    YCBmodel_directives = """
directives:
"""
    for i in range(2):
        object_num = rng.integers(0, len(ycb))
        if "cracker_box" in ycb[object_num]:
            # skip it. it's just too big!
            continue
        YCBmodel_directives += f"""
- add_model:
    name: ycb{i}
    file: package://manipulation/hydro/{ycb[object_num]}
"""
        
    YCBmodel_directives = (
        """
directives:
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
                
    - add_model:
        name: meatCan
        file: package://manipulation/hydro/010_potted_meat_can.sdf
        default_free_body_pose:
            base_link_meat:
                translation: [-0.05, -0.45, 0.05]
                
    - add_model:
        name: pickObject
        file: package://manipulation/hydro/004_sugar_box.sdf
        default_free_body_pose:
            base_link_sugar:
                translation: [0.05, -0.55, 0.05]
                rotation: !Rpy { deg: [0.0, 90.0, 0.0 ]}
    """
    )

    builder = DiagramBuilder()

    scenario = LoadScenario(
        filename=FindResource("models/clutter.scenarios.yaml"),
        scenario_name="Clutter",
    )
    scenario = AppendDirectives(scenario, data=YCBmodel_directives)
    '''
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=True, use_hardware=False
        ),
    )
    '''

    station = builder.AddSystem(MakeHardwareStation(scenario, mesh))

    
    plant = station.GetSubsystemByName("plant")
        
    robot = plant.GetModelInstanceByName("iiwa")

    #gripper = plant.GetModelInstanceByName("gripper")
    #end_effector_body = plant.GetBodyByName("body", gripper)
       
    #AddRgbdSensors(builder, plant, scene_graph)
    
    iiwa_controller = builder.AddSystem(TrajPosOut(plant))
    iiwa_controller.set_name("iiwa_controller")
    
    wsg_controller = builder.AddSystem(WSGOut(plant))
    wsg_controller.set_name("wsg_controller")

    to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder)

    y_bin_grasp_selector = builder.AddSystem(
        GraspSelector(
            plant,
            plant.GetModelInstanceByName("bin0"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
            ],
        )
    )

    AntipHandler = builder.AddSystem(AntipHandler())



    builder.Connect(
        to_point_cloud["camera0"].get_output_port(),
        y_bin_grasp_selector.get_input_port(0),
    )
    builder.Connect(
        to_point_cloud["camera1"].get_output_port(),
        y_bin_grasp_selector.get_input_port(1),
    )
    builder.Connect(
        to_point_cloud["camera2"].get_output_port(),
        y_bin_grasp_selector.get_input_port(2),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        y_bin_grasp_selector.GetInputPort("body_poses"),
    )

    builder.Connect(y_bin_grasp_selector.GetOutputPort("grasp_selection"), AntipHandler.GetInputPort("grasps"))

    builder.Connect(AntipHandler.GetOutputPort("grasp_position"), iiwa_controller.GetInputPort("grasps"))

    builder.Connect(AntipHandler.GetOutputPort("grasp_cost"), iiwa_controller.GetInputPort("grasp_cost"))
    
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), iiwa_controller.GetInputPort("iiwa_measured"))
    
    builder.Connect(wsg_controller.GetOutputPort("wsg_position"), station.GetInputPort("wsg.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("iiwa_position"), station.GetInputPort("iiwa.position"),)
    
    builder.Connect(iiwa_controller.GetOutputPort("plan_stage"), wsg_controller.GetInputPort("statusTime"))

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    
        
    return diagram, context, robot, plant, mesh

def LoadRobotHardwareStationHardware(builder = None):
    if not builder:
        builder = DiagramBuilder()
    
    model_directives = """    
directives:
- add_directives:
    file: package://manipulation/two_bins.dmd.yaml

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