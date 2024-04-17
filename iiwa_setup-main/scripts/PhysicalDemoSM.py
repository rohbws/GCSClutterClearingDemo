import numpy as np
from pydrake.all import Simulator
from pydrake.all import *
import queue
from pydrake.all import BasicVector
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.geometry.optimization import Point, GraphOfConvexSetsOptions
from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Simulator,
)
import time
from collections import OrderedDict


from pydrake.geometry.optimization import (
    LoadIrisRegionsYamlFile,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.framework import LeafSystem
import math

from pydrake.all import Simulator

from forward_kinematics import forward_kinematics, IiwaForwardKinematics

from HardwareSetup import *

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

        if self.runToStartRequried():
            startDist = np.linalg.norm(self.getRunStartPos(context) - self.lastPos)
            if startDist < 0.01:
                self.startReached = True
        else:
            self.startReached = True

        if self.firstCall and self.startReached:
            plantCont = self.plant.CreateDefaultContext()
            
            self.plant.SetPositions(plantCont, np.concatenate((self.lastPos,[0,0])))
            self.startEE = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()

            self.stackTime = context.get_time()

            self.firstCall = False
        

        if self.runToStartRequried() and not self.startReached:
            targPos = 0
            targPosCoeff = 0.9995
            targPosCoeff = math.atan(100.0*startDist) * 2.0 * 0.9995 * 0.3 / math.pi + 0.7
            targPos = targPosCoeff * np.array(self.lastPos).flatten() + (1-targPosCoeff) * np.array(self.getRunStartPos(context)).flatten()

            self.lastPos = targPos.copy()

            return targPos

        if self.type == 0 or self.type == 4:
            newPos = np.array(self.traj.value((context.get_time() - self.stackTime) * 0.2)).flatten()
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
        #newPos[0] = newPos[0] + 1.57
        return newPos

    def runToStartRequried(self) -> bool:
        return (self.type == 0 or self.type == 4)
    
    def getDistToStart(self, context) -> float:
        if self.runToStartRequried():
            return np.linalg.norm(self.getRunStartPos(context) - self.lastPos)

    def getRunStartPos(self, context):
        if self.runToStartRequried():
            if self.type == 0 or self.type == 4:
                return np.array(self.traj.value(0)).flatten()
            else:
                raise TypeError("Run to start not required for this stage")
            
    def exitConditionReached(self, context, exitThresh = 0.01) -> bool:
        if self.type == 0 or self.type == 4:
            endPos = np.array(self.traj.value(self.traj.end_time())).flatten()
            return np.linalg.norm(endPos - self.lastPos) < exitThresh
        elif self.type == 1:
            plantCont = self.plant.CreateDefaultContext()
            self.plant.SetPositions(plantCont, np.concatenate((self.lastPos,[0,0])))
            endPos = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()
            print(endPos[2] - self.startEE[2] + self.traj[0])
            return (endPos[2] - self.startEE[2] + self.traj[0]) < 0
        elif self.type == 3:
            plantCont = self.plant.CreateDefaultContext()
            self.plant.SetPositions(plantCont, np.concatenate((self.lastPos,[0,0])))
            endPos = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()
            print((endPos[2] - self.startEE[2] - self.traj[0]) )
            return (endPos[2] - self.startEE[2] - self.traj[0]) > 0
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
        self.gcsPlan = GCSPlanner()
        self.antGrasp = AntipodalGrasps()

        self.gcsPlan.addTrajEndPoints(seeds["Transition"], self.antGrasp.getNextGrasp())

        self.gcsPlan.planNextTraj()

        self.intermediates = queue.Queue()

        self.firstRun = True
        
                                     
    def getI(self, context, output):
        output.SetFromVector([self.state])

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
        
        output.SetFromVector(self.currTraj.getOutputPos(context))

        if self.currTraj.exitConditionReached(context):
            self.lastPos = self.currTraj.lastPos
            self.state = (self.state + 1) % 6
            if self.state == 0 or self.state == 4:
                self.currTraj = trajOperator(self.plant, self.state, self.gcsPlan.getNextTraj(), context, self.iiwaPos, self.lastPos)
            elif self.state == 1 or self.state == 3:
                self.currTraj = trajOperator(self.plant, self.state, [0.1], context, self.iiwaPos, self.lastPos)
            else:
                self.currTraj = trajOperator(self.plant, self.state, [2], context, self.iiwaPos, self.lastPos)
            self.outTraj += 1

        


class AntipodalGrasps():
    def __init__(self) -> None:
        self.graspPoses = queue.Queue()
        self.graspPoses.put([-2.28218173, 0.36183194, 0.6409329, -1.87001069, -0.25713113, 0.98730485, 0.0965661])
    
    def getNextGrasp(self):
        return self.graspPoses.get()
    

class GCSPlanner():
    def __init__(self):
        self.plant, diagram = LoadRobotHardwareStation()
        self.plant_context = self.plant.CreateDefaultContext()
        self.trajs = queue.Queue()
        self.trajInps = queue.Queue()
        iris_filename = "my_iris.yaml"
        self.iris_regions = dict()
        self.iris_regions.update(LoadIrisRegionsYamlFile(iris_filename))
        self.pruneRegions()

    def pruneRegions(self):
        del self.iris_regions["GraspPos12"]
        del self.iris_regions["GraspPos13"]
        del self.iris_regions["GraspPos17"]
        del self.iris_regions["GraspPos2"]
        del self.iris_regions["GraspPos22"]
        del self.iris_regions["GraspPos24"]


        #The below regions have never been utilized in an antipodal grasp
        del self.iris_regions["GraspPos10"]
        del self.iris_regions["GraspPos18"]
        del self.iris_regions["GraspPos19"]
        del self.iris_regions["GraspPos20"]
        del self.iris_regions["GraspPos21"]
        del self.iris_regions["GraspPos3"]
        del self.iris_regions["GraspPos4"]
        del self.iris_regions["GraspPos5"]
        #del iris_regions["GraspPos6"]
        del self.iris_regions["GraspPos7"]
        del self.iris_regions["GraspPos8"]
        del self.iris_regions["GraspPos9"]
        del self.iris_regions["GraspPos15"]
        del self.iris_regions["Transition"]
        del self.iris_regions["TransitionNoObs"]
        del self.iris_regions["TransitionNoObs2"]

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

diagram, context, robot, plant, internalPlantContext = make_environment_model_display(
    rng=np.random.default_rng(), num_ycb_objects=1, draw=True
)
wsg = plant.GetModelInstanceByName("wsg")

simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)
simulator.AdvanceTo(600.0)
