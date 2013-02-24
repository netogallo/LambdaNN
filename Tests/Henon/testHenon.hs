import Reservoir.Learning
import Reservoir.LearningUtils (networksProfiler,defaultError)
import Reservoir.Reservoir
import Data.Packed.Vector
import Foreign.Storable (Storable)
import Foreign.Storable.Tuple
import Data.Packed.Matrix

henonMapR sx _ a b 1 = sx
henonMapR sx sy a b n = (henonMapY (n-1)) + 1 - a*((henonMapR sx sy a b (n-1))^2)
  where
    henonMapY 1 = sy
    henonMapY n' = b * (henonMapR sx sy a b (n'-1))
    
henonMap sx sy a b n = let
  (res,_) = foldl henonMap' (0,fromList [0]) [1 .. n]
  in
   res
  where
    henonMap' (_,_) 1 = (sx,fromList [sx])
    henonMap' (_,_) 2 = let
      res = sy + 1 - a*(sx^2)
      in
       (res, fromList [sx,res])
    henonMap' (res,prev) val = let
      res = b*(prev @> 0) + 1 - a*((prev @> 1)^2)
      in
       (res,fromList [prev @> 1,res])
       
inVals :: [Vector Double]
inVals = [fromList [x] | x <- [1 .. ]]

hVals = map (\x -> mapVector (henonMap 1 1 1.4 0.3) x) inVals

adjTanh x = tanh x

config = (defaultConfig 0 100 1){inputFunction = mapVector adjTanh,
                                 inputMatrixRange = Continuous (-0.1,0.1),
                                 internalMatrixRange = Continuous (-0.1,0.1),
                                 outputFeedbackRange = Continuous (-0.1,0.1),
                                 internalSpectRadius = 0.95
                                }

initSize = 300
trainSize = 1000
testSize = 100

initial = take initSize dummy
initialOut = take initSize hVals
trainIn = take trainSize $ drop initSize dummy
trainOut = take trainSize $ drop initSize hVals
testIn = take testSize $ drop (initSize + trainSize) dummy
testOut = take testSize $ drop (initSize + trainSize) hVals
dummy = repeat $ fromList [0]

trainHenon tradeoff = do
  _ <- runNetworkTeacherForced initial initialOut
  --networkTrainerPInv trainIn trainOut
  networkTrainerRRegression tradeoff trainIn trainOut
  
hLearner = do
  reservoir <- makeReservoir config
  return $ runReservoirNoiseless reservoir $ trainHenon 0.1

hTester = do
  (_,reservoir) <- hLearner
  return $ runReservoirNoiseless reservoir $ runNetworkCollected testIn

hMaker = do
  reservoirs <- mapM (\_ -> makeReservoir config) [5..40]
  let
    networks = [network {internalWeights = mapMatrix (\e -> (s1/100)*e) (internalWeights network) } | s1 <- [1 .. 175] :: [Double],network <- reservoirs]
    rregressions = [networkTrainerRRegression (tradeoff/10) | tradeoff <- [1 .. 5] :: [Double]]
  let
    result = networksProfiler runWrapper networks errorMeasure rregressions dummy hVals 100 500 100
  return $ result
  where
    runWrapper :: Reservoir Double -> RunReservoirM Double Noiseless (Reservoir Double,Double) -> (RunningState Noiseless Double,(Reservoir Double,Double))
    runWrapper = runReservoirNoiseless
    errorMeasure :: Vector Double -> Vector Double -> Double
    errorMeasure v1s v2s = let 
      vector :: Vector (Double,Double)
      vector = zipVector v1s v2s
      in
       foldVector (\(v1,v2) error -> error + defaultError v1 v2) 0 vector
    
-- henonMapRec s a b p 0 = s
-- henonMapRec s a b p 1 = 
-- henonMapRec s a b p n = henonMapY + 1 - a*((p ! 0)^2)
--   where
--     henonMapY = b * (p ! 1)

-- loadTimeSeries h = do
--   series <- hGetContents h >>= return . lines
--   s <- return $ (map read series :: [Double])
--   return s

-- inVals = [fromList [x] | x <- [0 .. 500]]

-- sinVals = unsafePerformIO $ do
--   h <- openFile "Tests/Henon/henon.dat" ReadMode
--   series <- loadTimeSeries h
--   vals <- return $  take 500 $ series
--   --hClose h
--   return $ map (\x -> fromList [x]) vals

-- initial = take 100 inVals
-- initialOut = take 100 sinVals
-- trainIn = take 1000 $ drop 100 inVals
-- trainOut = take 1000 $ drop 100 sinVals
-- testIn = take 100 $ drop 400 inVals
-- testOut = take 100 $ drop 400 sinVals

-- trainSin = do
--   _ <- runNetworkCollectedTeacherForced initial initialOut
--   networkTrainerPInv trainIn trainOut
  
-- sinLearner = do
--   reservoir <- makeReservoir 1 1 200 0.1
--   return $ runReservoirNoiseless reservoir funs trainSin

-- sinTester = do
--   (_,reservoir) <- sinLearner
--   return $ runReservoirNoiseless reservoir funs $ runNetworkCollected testIn

-- funs = (tan,atan)

