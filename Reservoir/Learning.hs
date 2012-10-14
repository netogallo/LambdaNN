{-# LANGUAGE FlexibleInstances #-}
module Reservoir.Learning where

import Control.Monad              (foldM)
import Data.Packed.Matrix
import Data.Packed.Vector
import Numeric.Container          ((<>))
import Numeric.LinearAlgebra.Algorithms (pinv)
import Numeric.LinearAlgebra.Util (zeros)
import Reservoir.Reservoir

data TrainingState a = TrainingState (Reservoir a) [Vector a] [Vector a]

instance Show (TrainingState Double) where
  show (TrainingState r v1 v2) = "TrainingState "++(show r)++" "++(show v1)++" "++(show v2)

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

updateNetwork noise (f,fInv) input (Reservoir state out inWM intWM outWM ofbWM) = let
  globalMatrix = fromColumns $ (toColumns inWM) ++ (toColumns intWM) ++ (toColumns ofbWM)
  globalState = fromColumns [join [input,state,out]]
  adjFun = head . toColumns
  newState = f $ noise + (adjFun $ globalMatrix <> globalState)
  newGlobalState = fromColumns [join [input,newState]]
  newOut = f $ adjFun $ outWM <> newGlobalState
  in
   Reservoir newState newOut inWM intWM outWM ofbWM


