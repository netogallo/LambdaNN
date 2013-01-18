import Reservoir.Learning
import Reservoir.Reservoir
import Data.Packed.Vector

inVals = [fromList [x] | x <- [0 .. 500]]

sinVals = map (\x -> mapVector sin x) inVals

initial = take 100 inVals
trainIn = take 300 $ drop 100 inVals
trainOut = take 300 $ drop 100 sinVals
testIn = take 100 $ drop 400 inVals
testOut = take 100 $ drop 400 sinVals

trainSin = do
  _ <- runNetwork initial
  networkTrainerPInv trainIn trainOut
  
sinLearner = do
  reservoir <- makeReservoir 1 1 50 0.2
  return $ runReservoirNoiseless reservoir (id,id) trainSin

sinTester = do
  (_,reservoir) <- sinLearner
  return $ runReservoirNoiseless reservoir (id,id) $ runNetworkCollected testIn
  
