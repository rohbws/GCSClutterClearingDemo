import numpy as np
from pydrake.all import Simulator, IrisInConfigurationSpaceFromCliqueCover
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
from manipulation.clutter import GenerateAntipodalGraspCandidate

from forward_kinematics import forward_kinematics, IiwaForwardKinematics

from HardwareSetup import *

import HardwareSetup as HS

from manipulation import ConfigureParser, FindResource, running_as_notebook


iris_filename = "my_irisSim.yaml"
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
        self.plantInstance = self.plant.GetModelInstanceByName("iiwa")
        self.firstCall = True
        self.startReached = False
        self.stackTime = context.get_time()
        self.lastPos = self.getiiwaPos(context)

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
            
            self.plant.SetPositions(plantCont, self.plantInstance, self.lastPos)
            self.startEE = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()

            self.stackTime = context.get_time()

            self.firstCall = False
        

        if self.runToStartRequried() and not self.startReached:
            targPos = 0
            targPosCoeff = 0.9995
            targPosCoeff = math.atan(1000.0*startDist) * 2.0 * 0.9995 * 0.3 / math.pi + 0.7
            targPos = targPosCoeff * np.array(self.lastPos).flatten() + (1-targPosCoeff) * np.array(self.getRunStartPos(context)).flatten()

            self.lastPos = targPos.copy()

            return targPos

        if self.type == 0 or self.type == 4:
            newPos = np.array(self.traj.value((context.get_time() - self.stackTime) * 0.2)).flatten()
            #newPos[0] = newPos[0] + 1.57
        elif self.type == 1:
            plantCont = self.plant.CreateDefaultContext()
            curPos = self.lastPos.copy()
            #curPos = np.concatenate((curPos, [0,0]))
            self.plant.SetPositions(plantCont, self.plantInstance, curPos)

            params = DifferentialInverseKinematicsParameters(self.plant.num_positions(),
                                                             self.plant.num_velocities())
            newPos = np.array(DoDifferentialInverseKinematics(self.plant, plantCont, [0, 0, 0, 0, 0,-1], self.plant.GetFrameByName("body", self.gripper), params).joint_velocities)[0:7]
            newPos = np.array(curPos)[0:7] + 0.0001 * newPos
        elif self.type == 3:
            plantCont = self.plant.CreateDefaultContext()
            curPos = self.lastPos.copy()
            #curPos = np.concatenate((curPos, [0,0]))
            self.plant.SetPositions(plantCont, self.plantInstance, curPos)

            params = DifferentialInverseKinematicsParameters(self.plant.num_positions(),
                                                             self.plant.num_velocities())
            newPos = np.array(DoDifferentialInverseKinematics(self.plant, plantCont, [0, 0, 0, 0, 0,1], self.plant.GetFrameByName("body", self.gripper), params).joint_velocities)[0:7]
            newPos = np.array(curPos)[0:7] + 0.0001 * newPos
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
                returnVal = np.array(self.traj.value(0)).flatten()
                #returnVal[0] += 1.57
                return returnVal
            else:
                raise TypeError("Run to start not required for this stage")
            
    def exitConditionReached(self, context, exitThresh = 0.01) -> bool:
        if self.type == 0 or self.type == 4:
            endPos = np.array(self.traj.value(self.traj.end_time())).flatten()
            #endPos[0] += 1.57
            return np.linalg.norm(endPos - self.lastPos) < exitThresh
        elif self.type == 1:
            plantCont = self.plant.CreateDefaultContext()
            self.plant.SetPositions(plantCont, self.plantInstance, self.lastPos)
            endPos = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()
            #print(endPos[2] - self.startEE[2] + self.traj[0])
            return (endPos[2] - self.startEE[2] + self.traj[0]) < 0
        elif self.type == 3:
            plantCont = self.plant.CreateDefaultContext()
            self.plant.SetPositions(plantCont, self.plantInstance, self.lastPos)
            endPos = self.plant.GetFrameByName("body", self.gripper).CalcPoseInWorld(plantCont).translation()
            #print((endPos[2] - self.startEE[2] - self.traj[0]) )
            return (endPos[2] - self.startEE[2] - self.traj[0]) > 0
        elif self.type == 2 or self.type == 5:
            return context.get_time() - self.stackTime - self.traj[0] > 0
        elif self.type == 5.5:
            return not (trajsEmpty(self.traj[0]))

        
def planNextTraj(trajInps, gcsTrajs, planner):
        nextTraj = trajInps.get()
        gcsTrajs.put(planner.GcsTrajOpt(nextTraj[0], nextTraj[1]))

from pydrake.trajectories import BezierCurve, CompositeTrajectory

def getNextTraj(gcstrajs):
        if gcstrajs.empty():
            return None
        else:
            nextTraj = gcstrajs.get()
            for i in range(len(nextTraj)):
                nextTraj[i] = BezierCurve(nextTraj[i][0], nextTraj[i][1], nextTraj[i][2])
            return CompositeTrajectory(nextTraj)
        
def trajsEmpty(gcstrajs):
    return gcstrajs.empty()
        
def planTrajs(trajInps, gcsTrajs, planner):
    while True:
        if trajInps.empty():
            #print("Waiting for next traj inps")
            time.sleep(0.1)
        else:
            print("planning!")
            planNextTraj(trajInps, gcsTrajs, planner)

# Takes 3 point clouds (in world coordinates) as input, and outputs and estimated pose for the mustard bottle.
class GraspSelector(LeafSystem):
    def __init__(self, plant, bin_instance, camera_body_indices):
        LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make((np.inf, RigidTransform())),
            self.SelectGrasp,
        )
        port.disable_caching_by_default()

        # Compute crop box.
        context = plant.CreateDefaultContext()
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        a = X_B.multiply(
            [-0.22 + 0.025 + margin, -0.29 + 0.025 + margin, 0.015 + margin]
        )
        b = X_B.multiply([0.22 - 0.1 - margin, 0.29 - 0.025 - margin, 2.0])
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        self._internal_model = HS.make_internal_model()
        self._internal_model_context = self._internal_model.CreateDefaultContext()
        self._rng = np.random.default_rng()
        self._camera_body_indices = camera_body_indices

    def SelectGrasp(self, context, output):
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)
            pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())
        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)

        costs = []
        X_Gs = []
        # TODO(russt): Take the randomness from an input port, and re-enable
        # caching.
        for i in range(100):
            cost, X_G = GenerateAntipodalGraspCandidate(
                self._internal_model,
                self._internal_model_context,
                down_sampled_pcd,
                self._rng,
            )
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if len(costs) == 0:
            # Didn't find a viable grasp candidate
            X_WG = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2), [0.5, 0, 0.22]
            )
            print("NO VIABLE GRASP")
            output.set_value((np.inf, X_WG))
        else:
            best = np.argmin(costs)
            output.set_value((costs[best], X_Gs[best]))

import multiprocessing

class TrajPosOut(LeafSystem):

    def __init__(self, plant, plant_context=None):
        LeafSystem.__init__(self)
        self.traje = []
        self.plant = plant
        self.plant_context = plant_context

        self.iiwaPos = self.DeclareVectorInputPort("iiwa_measured", BasicVector(7))
        self.grasps = self.DeclareVectorInputPort("grasps", BasicVector(7))
        self.graspCost = self.DeclareVectorInputPort("grasp_cost", BasicVector(1))
        
        self.DeclareVectorOutputPort("iiwa_position", BasicVector(7), self.calciiwaPos)
        self.DeclareVectorOutputPort("plan_stage", BasicVector(1), self.getI)

        self.state = 0
        self.outTraj = 0
        self.stackTime = 0
        self.lastPos = [None]
        self.gcsPlan = GCSPlanner()
        self.antGrasp = AntipodalGrasps()

        manager = multiprocessing.Manager()
        # Create a shared queue
        self.gcsTrajs = manager.Queue()

        self.trajInps = manager.Queue()

        #initialize parallel process that runs self.gcsPlan.planNextTraj() in the background
        self.backPlanning = multiprocessing.Process(target=planTrajs, args=(self.trajInps, self.gcsTrajs, self.gcsPlan))

        self.backPlanning.daemon = True

        self.trajInps.put([seeds["Transition"], [-2.27218173, 0.37183194, 0.6459329, -1.86001069, -0.26713113, 0.97730485, 0.1065661]])
        self.trajInps.put([seeds["Above Bin 1"], seeds["Above Bin 2"]])


        self.currTraj = None

        self.firstRun = True

        self.depRun = None

        self.callGraspFinding = True
        
                                     
    def getI(self, context, output):
        if self.state == 5.5:
            output.SetFromVector([self.prevState])
        else:
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
        5.5 - Awaiting new traj


        Within each execution:
        if startPos is not None run to start position
        execute command
        - run until traj class determines execution conditions met
        - - determined purely based on context (time or position)
        repeat
        '''

        if self.state == 5 and self.callGraspFinding:
            grasp = self.grasps.Eval(context)
            print(self.graspCost.Eval(context)[0])
            while self.graspCost.Eval(context)[0] == np.inf:
                print("TESTT")
                grasp = self.grasps.Eval(context)
                print(self.graspCost.Eval(context)[0])
            self.trajInps.put([seeds["Transition"], grasp])
            self.trajInps.put([grasp, seeds["Above Bin 2"]])
            self.callGraspFinding = False
        elif not self.state == 5:
            self.callGraspFinding= True
        

        if not self.lastPos[0]:
            self.lastPos = self.iiwaPos.Eval(context)

        if self.firstRun:
            self.backPlanning.start()
        
            self.firstRun = False

        if self.currTraj is None:
            output.SetFromVector(self.lastPos)
            newTraj = getNextTraj(self.gcsTrajs)
            if newTraj is None:
                self.currTraj = None
            else:
                self.currTraj = trajOperator(self.plant, self.state, newTraj, context, self.iiwaPos, self.lastPos)
        else:
            outPos = self.currTraj.getOutputPos(context)
                
            output.SetFromVector(outPos)

            self.lastPos = outPos
            

            if self.currTraj.exitConditionReached(context):
                self.lastPos = self.currTraj.lastPos 

                if self.state == 5.5:
                    self.state = self.prevState
                else:
                    self.state = (self.state + 1) % 6

                print("State " + str(self.state))
                if self.state == 0 or self.state == 4:
                    newTraj = getNextTraj(self.gcsTrajs)
                    if newTraj is None:
                        self.prevState = self.state
                        self.state = 5.5
                        self.currTraj = trajOperator(self.plant, self.state, [self.gcsTrajs], context, self.iiwaPos, self.lastPos)
                    else:
                        self.currTraj = trajOperator(self.plant, self.state, newTraj, context, self.iiwaPos, self.lastPos)
                elif self.state == 1 or self.state == 3:
                    self.currTraj = trajOperator(self.plant, self.state, [0.2], context, self.iiwaPos, self.lastPos)
                else:
                    self.currTraj = trajOperator(self.plant, self.state, [2], context, self.iiwaPos, self.lastPos)
                self.outTraj += 1

        
#First grasp: GraspPos0
#Second grasp: GraspPos27


class AntipodalGrasps(LeafSystem):
    def __init__(self) -> None:
        LeafSystem.__init__(self)
        self.antipWSG = self.DeclareAbstractInputPort(
            "grasps", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()

        self.plant = HS.LoadRobotHardwareStation()[0]
        self.plant_context = self.plant.CreateDefaultContext()

        self.DeclareVectorOutputPort("grasp_position", BasicVector(7), self.getNextGrasp)
        self.DeclareVectorOutputPort("grasp_cost", BasicVector(1), self.getCost)

        '''
        self.graspPoses = queue.Queue()
        self.graspPoses.put([-2.27218173, 0.37183194, 0.6459329, -1.86001069, -0.26713113, 0.97730485, 0.1065661])
        self.graspPoses.put([-1.9933341, 0.58914925, 0.62913132, -1.55970203, -0.37290568, 1.09890311, 0.1065661])
        self.graspPoses.put([-2.29218173, 0.35183194, 0.6359329, -1.88001069, -0.24713113, 0.99730485, 0.0865661])
        self.graspPoses.put([-1.9933341, 0.56914925, 0.60913132, -1.57970203, -0.35290568, 1.11890311, 0.0865661])
        self.graspPoses.put([-2.29218173, 0.37183194, 0.6459329, -1.86001069, -0.26713113, 0.97730485, 0.1065661])
        self.graspPoses.put([-2.0133341, 0.58914925, 0.62913132, -1.55970203, -0.37290568, 1.09890311, 0.1065661])
        self.graspPoses.put([-2.27218173, 0.35183194, 0.6359329, -1.88001069, -0.24713113, 0.99730485, 0.0865661])
        self.graspPoses.put([-1.9933341, 0.56914925, 0.61913132, -1.57970203, -0.36290568, 1.11890311, 0.0965661])
        self.graspPoses.put([-2.29218173, 0.36183194, 0.6409329, -1.87001069, -0.25713113, 0.98730485, 0.0965661])
        self.graspPoses.put([-2.0033341, 0.57914925, 0.62913132, -1.56970203, -0.36290568, 1.10890311, 0.0965661])
        self.graspPoses.put([-2.28218173, 0.37183194, 0.6459329, -1.86001069, -0.26713113, 0.97730485, 0.1065661])
        self.graspPoses.put([-2.0133341, 0.58914925, 0.61913132, -1.55970203, -0.37290568, 1.09890311, 0.1065661])
        '''


    
    def getNextGrasp(self, context, output):
        
        #return self.graspPoses.get()

        self.cost, X_G = self.get_input_port(self.antipWSG).Eval(
                    context
                )
        
        trans = X_G.translation().copy()
        print(trans)
        trans[2] += 0.1

        X_G.set_translation(trans)
        
        iiwaPos = HS.MyInverseKinematics(X_G, self.plant, self.plant_context, seeds["Above Bin 1"])

        output.SetFromVector(iiwaPos)

    def getCost(self, context, output):
        print(self.cost)
        output.SetFromVector([self.cost])
    

class GCSPlanner():
    def __init__(self):
        self.plant, diagram = LoadRobotHardwareStation()
        self.plant_context = self.plant.CreateDefaultContext()
        #self.trajs = queue.Queue()
        #self.trajInps = queue.Queue()
        iris_filename = "my_irisSim.yaml"
        self.grasPosRegs = [['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos26', 'GraspPos27', 'GraspPos6'], ['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos26', 'GraspPos27', 'GraspPos6'], ['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos0', 'GraspPos6'], ['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos26', 'GraspPos27', 'GraspPos6'], ['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos26', 'GraspPos27', 'GraspPos6'], ['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos26', 'GraspPos27', 'GraspPos6'], ['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos0', 'GraspPos6'], ['GraspPos1', 'GraspPos11', 'GraspPos14', 'GraspPos16', 'GraspPos25', 'GraspPos26', 'GraspPos27', 'GraspPos6']]
        self.iris_regions = dict()
        self.iris_regions.update(LoadIrisRegionsYamlFile(iris_filename))
        self.iter = 0
        #self.pruneRegions()

    def pruneSelectRegions(self, names):
        for name in names:
            del self.iris_regions[name]

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
        self.iris_regions = dict()
        self.iris_regions.update(LoadIrisRegionsYamlFile(iris_filename))
        #self.pruneRegions()
        self.iter += 1
        if not self.iris_regions:
            print(
                "No IRIS regions loaded. Make some IRIS regions then come back and try this again."
            )
            return
        
        print(f"Planning trajectory {self.iter} from {q_start} to {q_goal}")
        assert len(q_start) == len(q_goal)
        assert len(q_start) == self.iris_regions[next(iter(self.iris_regions))].ambient_dimension()

        edges = []

        gcs = GcsTrajectoryOptimization(len(q_start))
        regions = gcs.AddRegions(list(self.iris_regions.values()), order=1)
        source = gcs.AddRegions([Point(q_start)], order=0)
        target = gcs.AddRegions([Point(q_goal)], order=0)
        edges.append(gcs.AddEdges(source, regions))
        edges.append(gcs.AddEdges(regions, target))

        
        gcs.AddTimeCost(weight=0.5)
        gcs.AddPathLengthCost()
        gcs.AddVelocityBounds(
            self.plant.GetVelocityLowerLimits(), (self.plant.GetVelocityUpperLimits())
        )

        options = GraphOfConvexSetsOptions()
        options.solver = MosekSolver()
        options.preprocessing = True
        #options.max_rounded_paths = 5
        start_time = time.time()
        print("Running GCS")
        traj, result = gcs.SolvePath(source, target, options)
        print(f"GCS solved in {time.time() - start_time} seconds")
        if not result.is_success():
            print("Could not find a feasible path from q_start to q_goal")
            return
        
        trajList = []
        
        for i in range(traj.get_number_of_segments()):
            trajTemp = traj.segment(i)
            trajList.append([trajTemp.start_time(), trajTemp.end_time(), trajTemp.control_points()])
        
        return trajList

class WSGOut(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("wsg_position", BasicVector(1), self.calciiwaPos)
        self.statusTime = self.DeclareVectorInputPort("statusTime", 1)
        self.closeTimes = [2., 3., 4.]
        
    def calciiwaPos(self, context, output):

        #print(self.statusTime.Eval(context))

        if self.statusTime.Eval(context)[0]  == 2.0 or self.statusTime.Eval(context)[0]  == 3.0 or self.statusTime.Eval(context)[0]  == 4.0:
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

diagram, context, robot, plant, mesh = make_environment_model_sim(
    TrajPosOut=TrajPosOut, WSGOut=WSGOut, GraspSelector = GraspSelector, AntipHandler = AntipodalGrasps
)
wsg = plant.GetModelInstanceByName("wsg")

simulator = Simulator(diagram)
#simulator.AdvanceTo(0.1)
#mesh.Flush()
#print("Running sim")
simulator.Initialize()

mesh.Flush()

mesh.StartRecording()
simulator.set_target_realtime_rate(10)

try:
    simulator.AdvanceTo(300.0)
except:
    ...

mesh.StopRecording()
mesh.PublishRecording()
time.sleep(50)