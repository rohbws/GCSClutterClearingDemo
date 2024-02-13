import argparse

import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import load_scenario
from pydrake.all import DiagramBuilder, MeshcatPoseSliders, MeshcatVisualizer, Simulator

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    MeshcatPoseSliders,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    RandomGenerator,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
    PiecewisePolynomial,
    PiecewisePose,
    Quaternion,
)
import math

class TrajPosOut(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.startPos = self.DeclareAbstractInputPort("currPos", AbstractValue.Make(RigidTransform()))
        self.finPos = self.DeclareAbstractInputPort("finPos", AbstractValue.Make(RigidTransform()))
        self.currPos = self.DeclareVectorInputPort("currIiwaPos", 7)
        self.DeclareAbstractOutputPort("iiwaPos", lambda: AbstractValue.Make(RigidTransform()), self.calciiwaPos)
    
    def calciiwaPos(self, context, output):
        print(self.currPos.Eval(context))
        startposCart = self.startPos.Eval(context).translation()
        stopPoscart = self.finPos.Eval(context).translation()
        stopPoscart = [stopPoscart[0], stopPoscart[1], stopPoscart[2] - 0.127]
        self.finPosCopy = RigidTransform(self.finPos.Eval(context).rotation(), stopPoscart)
        dist = 10 * math.sqrt((startposCart[0] - stopPoscart[0]) ** 2 + (startposCart[1] - stopPoscart[1]) ** 2 + (startposCart[2] - stopPoscart[2]) ** 2)
        linMoveTraj = PiecewisePose.MakeLinear([context.get_time(), context.get_time() + dist], [self.startPos.Eval(context), self.finPosCopy])
        linMoveTrajPos = linMoveTraj.get_position_trajectory()
        linMoveTrajAng = linMoveTraj.get_orientation_trajectory()
        X_G = RigidTransform(Quaternion(linMoveTrajAng.value(linMoveTraj.end_time())), linMoveTrajPos.value(linMoveTraj.end_time()))
        if (dist) > 0.15:
            X_G = RigidTransform(Quaternion(linMoveTrajAng.value(context.get_time() + 0.15)), linMoveTrajPos.value(context.get_time() + 0.15))        
        output.get_mutable_value().set(X_G.rotation(), X_G.translation())




def main(use_hardware: bool, has_wsg: bool) -> None:
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

    builder = DiagramBuilder()

    scenario = load_scenario(data=scenario_data)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=has_wsg, use_hardware=use_hardware
        ),
    )

    slowingFunction = builder.AddSystem(TrajPosOut())

    iiwa_forward_kinematics = builder.AddSystem(
        IiwaForwardKinematics(station.get_internal_plant())
    )

    # Set up teleop widgets
    teleop = builder.AddSystem(
        MeshcatPoseSliders(
            station.internal_meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
        )
    )
    controller_plant = station.get_iiwa_controller_plant()


    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName("iiwa_link_7"),
    )

    builder.Connect(iiwa_forward_kinematics.get_output_port(), slowingFunction.GetInputPort("currPos"))
    builder.Connect(teleop.get_output_port(), slowingFunction.GetInputPort("finPos"))

    builder.Connect(slowingFunction.get_output_port(), differential_ik.GetInputPort("X_WE_desired"))

    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    builder.Connect(differential_ik.get_output_port(), slowingFunction.GetInputPort("currIiwaPos"))

    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        differential_ik.GetInputPort("robot_state"),
    )

    '''
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    '''

    builder.Connect(
        station.GetOutputPort("iiwa.position_commanded"),
        iiwa_forward_kinematics.get_input_port(),
    )
    builder.Connect(iiwa_forward_kinematics.get_output_port(), teleop.get_input_port())
    if has_wsg:
        wsg_teleop = builder.AddSystem(WsgButton(station.internal_meshcat))
        builder.Connect(
            wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
        )

    # Required for visualizing the internal station
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()

    controller_plant_context = controller_plant.CreateDefaultContext()

    controller_plant.SetPositions(controller_plant_context, [-1.57, 0.3, 0, -1.8, 0, 1, 1.57])

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
    station.internal_meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--has_wsg",
        action="store_true",
        help="Whether the iiwa has a WSG gripper or not.",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware, has_wsg=args.has_wsg)
