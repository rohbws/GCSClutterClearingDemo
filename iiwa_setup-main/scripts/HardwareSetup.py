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

from manipulation.station import load_scenario
from pydrake.all import DiagramBuilder
from iiwa_setup.iiwa import IiwaHardwareStationDiagram


def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def make_environment_model_display(
    has_wsg = True, hardware = False
) -> Body:
    # Make one model of the environment, but the robot only gets to see the sensor outputs.

    builder = DiagramBuilder()
    
    
    scenario_data = (
        """
    directives:
    - add_directives:
        file: package://manipulation/two_bins.dmd.yaml
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
            
    """)

    scenario = load_scenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=True, use_hardware=hardware,
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

def LoadRobotHardwareStation(builder = None):
    if not builder:
        builder = DiagramBuilder()
    
    model_directives = (
        """
    directives:
    - add_directives:
        file: package://manipulation/two_bins.dmd.yaml
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
            
    """)

    scenario = load_scenario(data=model_directives)
    station = builder.AddSystem(MakeHardwareStation(scenario))
    
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