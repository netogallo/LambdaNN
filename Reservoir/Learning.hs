{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ExistentialQuantification #-}
module Reservoir.Learning where

import Control.Monad              (foldM,forM)
import Data.Packed.Matrix
import Data.Packed.Vector
import Numeric.Container          ((<>))
import Numeric.LinearAlgebra.Algorithms (pinv)
import Numeric.LinearAlgebra.Util (zeros)
import Reservoir.Reservoir
import System.Random
import Data.Maybe (fromJust)

data Noiseless = Noiseless

data RunningState rand a = RunningState {
     runtimeReservoir :: Reservoir a,
     noiseGenerator :: (rand,rand -> (a,rand)),
     networkFunctions :: (ReservoirState a -> ReservoirState a,ReservoirState a -> ReservoirState a)
}

data RunReservoirM context rand b = RunReservoirM {runReservoirM :: RunningState rand context -> (RunningState rand context,b)}

instance Monad (RunReservoirM context rand) where
         m >>= k = let
           bind runState = (\(runState',res) -> runReservoirM (k res) runState') $ runReservoirM m $ runState            
           in
            RunReservoirM bind
         return a = RunReservoirM $ \runState -> (runState,a)

instance Functor (RunReservoirM context rand) where
  fmap f a = a >>= return . f

data TrainingState a = TrainingState (Reservoir a) [Vector a] [Vector a]

instance Show (TrainingState Double) where
  show (TrainingState r v1 v2) = "TrainingState "++(show r)++" "++(show v1)++" "++(show v2)

runReservoirNoiseless reservoir functions reservoirM = (runReservoirM reservoirM) $ RunningState reservoir (Noiseless,noiseless) functions
                                                       where
                                                         noiseless _ = (0,Noiseless)
  

{-
trainNetwork startup (f,fInv) reservoir sequence noiseLevel = undefined

runNetworkIO noiseLv (f,fInv) reservoir sequence = do
  (seq,result) <- foldM updateFun ([],reservoir) sequence
  return (reverse seq,result)

  where
    updateFun (seq,res) input = do
      Reservoir state output inWM intWM outWM ofbWM <- updateNetworkIO noiseLv (f,fInv) input res
      return (output:seq,Reservoir state output inWM intWM outWM ofbWM)

runNetworkTrainingIO noiseLv startup (f,fInv) reservoir sequence = do
  warmNetwork <- warmupNetwork
  exposeResult <- foldM trainerFun (TrainingState warmNetwork [] []) trainSequence
  let
    TrainingState exposedNetwork stateCollect outCollect = exposeResult
    (Reservoir state out inWM intWM outWM ofbWM) = exposedNetwork
    newWM = trans $ (pinv $ fromRows stateCollect) <> (fromRows outCollect)
  return $ Reservoir state out inWM intWM newWM ofbWM

  where
    startSequence = take startup sequence
    trainSequence = drop startup sequence
    warmerFun reservoir' inTeach = trainUpdateNetworkIO noiseLv (f,fInv) inTeach reservoir'
    trainerFun trainingState inTeach = runTrainingStepIO noiseLv (f,fInv) inTeach trainingState
    warmupNetwork = foldM warmerFun reservoir startSequence
    

runTrainingStepIO noiseLv (f,fInv) (input,teach) trainingState = do
  TrainingState n state out <- runTrainingStepIO' noiseLv (f,fInv) (input,teach) trainingState
  return (TrainingState n (reverse state) (reverse out))

runTrainingStepIO' noiseLv (f,fInv) (input,teach) trainingState = do
  Reservoir state output inWM intWM outWM ofbWM <- networkUpdater
  let
    newState = Data.Packed.Vector.join [f input,state]
    newReservoir = Reservoir state output inWM intWM outWM ofbWM
  return $ TrainingState newReservoir (newState:stateCollect) ((fInv output):outputCollect)
  where
    networkUpdater = trainUpdateNetworkIO noiseLv (f,fInv) (input,teach) reservoir
    TrainingState reservoir stateCollect outputCollect = trainingState

trainUpdateNetworkIO noiseLv f inTeach reservoir = let
  (Reservoir state out inWM intWM outWM ofbWM) = reservoir
  stateDim = dim state
  noisyTrain = addNoise trainUpdateNetwork stateDim noiseLv
  in
   noisyTrain >>= \nt -> return $ nt f inTeach reservoir

updateNetworkIO noiseLv f input reservoir = let
  (Reservoir state out inWM intWM outWM ofbWM) = reservoir
  stateDim = dim state
  noisyUp = addNoise updateNetwork stateDim noiseLv
  in
   noisyUp >>= \nu -> return $ nu f input reservoir

addNoise f dims noiseLv = do
  noise <- rand dims 1 >>= return . head . toColumns >>= return . ((*) noiseLv) . ((-) 0.5)
  (return.f) noise

trainUpdateNetwork noise (f,fInv) (input,teach) reservoir = let
  Reservoir state output inWM intWM outWM ofbWM = updateNetwork noise (f,fInv) input reservoir
  in
   Reservoir state teach inWM intWM outWM ofbWM
-}

networkTrainerPInv inputs teach = do
  (intStates,_) <- runNetworkCollectedTeacherForced inputs teach
  reservoir <- getReservoir
  let
    teachMatrix = fromRows teach
    newOutWM = trans $ (pinv intStates) <> teachMatrix
  return $ Reservoir (internalState reservoir) (outputState reservoir) (inputWeights reservoir) (internalWeights reservoir) newOutWM (outputFeedbackWeights reservoir)
    

collectReservoirState reservoir oldReservoir input history =
  let        
    newIntState = asRow $ internalState reservoir
    newOutState = asRow $ outputState reservoir
  in
   case history of
     Nothing -> Just (fromBlocks [[asRow input,newIntState,asRow $ outputState oldReservoir]],newOutState)
     Just (states,outputs) -> Just (fromBlocks [[states],[asRow . join $ [input,internalState reservoir,outputState oldReservoir]]],fromBlocks [[outputs],[newOutState]])

runNetworkCollected timeSeries = foldM collectState Nothing timeSeries
  >>= return . fromJust
  where
    collectState history input = do
      oldReservoir <- getReservoir
      reservoir <- updateNetwork input
      return $ collectReservoirState reservoir oldReservoir input history

runNetworkCollectedTeacherForced inputs outputs = do 
  res <- foldM collectState Nothing $ zip inputs outputs
  return $ fromJust res
  where
    collectState history (input,output) = do
      oldReservoir <- getReservoir
      reservoir <- updateNetworkTeacherForced input output
      return $ collectReservoirState reservoir oldReservoir input history

runNetwork timeSeries = forM timeSeries (\value -> updateNetwork value >>= return . outputState)

updateState reservoir runningState = RunningState reservoir (noiseGenerator runningState) (networkFunctions runningState)

getNoise = RunReservoirM getNoiseM

getNoiseM runningState = 
  let
    (oldGenerator,noiseFun) = noiseGenerator runningState
    (val,generator) = noiseFun oldGenerator
  in
   (RunningState (runtimeReservoir runningState) (generator,noiseFun) (networkFunctions runningState),val)

getReservoir = RunReservoirM $ \runningState -> (runningState,runtimeReservoir runningState)

setReservoir reservoir = RunReservoirM $ \runningState -> (RunningState reservoir (noiseGenerator runningState) (networkFunctions runningState),())

updateNetworkTeacherForced input output = do
  reservoir <- updateNetwork input
  let
    newReservoir = updateReservoirState (internalState reservoir) output reservoir
  setReservoir newReservoir
  return newReservoir
  
updateNetwork input = do
  r <- getReservoir
  let
    null = zeros (dim $ internalState r) 1
  n <- mapMatrixWithIndexM (\_ _ -> getNoise) null >>= return . head . toColumns
  RunReservoirM $ updateNetworkM n input

updateNetworkM noise input runningState =
  let
    (f,fOut) = networkFunctions runningState
    reservoir = runtimeReservoir runningState
    --adjFun = head . toColumns
    newState = f $ noise + (inputWeights reservoir <> input) + (internalWeights reservoir <> internalState reservoir) + (outputFeedbackWeights reservoir <> outputState reservoir)
    -- globalMatrix = fromBlocks [[inputWeights reservoir,internalWeights reservoir,outputWeights reservoir]]
    -- globalState = fromColumns [join [input,internalState reservoir,outputState reservoir]]

    -- newState = f $ noise + (adjFun $ globalMatrix <> globalState)
    -- newGlobalState = fromColumns [join [input,newState]]
    newOut = fOut $ (outputWeights reservoir) <> (join [input,newState,outputState reservoir])
    newNetwork = updateReservoirState newState newOut reservoir
  in
   (updateState newNetwork runningState,newNetwork)

