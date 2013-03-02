import Reservoir.Learning
import Reservoir.LearningUtils 
import Reservoir.Reservoir
import Data.Packed.Vector
import Foreign.Storable (Storable)
import Foreign.Storable.Tuple
import Data.Packed.Matrix
import Graphics.Gnuplot.Simple

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
-- hVals = map (\x -> mapVector (henonMap 1 2 2 1) x) inVals
--hVals = map (mapVector sin) inVals

adjTanh x = tanh x

config = (defaultConfig 0 100 1){inputFunction = mapVector adjTanh,
                                 inputMatrixRange = Continuous (-4,4),
                                 internalMatrixRange = Continuous (-1,1),
                                 outputFeedbackRange = Continuous (-4,4),
                                 internalSpectRadius = 1.3
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
  return $ runReservoirNoiseless reservoir $ trainHenon 2.5

hTester = do
  (_,reservoir) <- hLearner
  (_,(reservoir,states,output)) <- return $ runReservoirNoiseless reservoir $ do 
    (states,out) <- runNetworkCollected testIn
    outReservoir <- getReservoir
    return (outReservoir,states,out)
  return (reservoir,states,output)

mainPlot = do
  (reservoir,states,output) <- hTester
  plotVectorsStyle [Title "Henon vs Output"] LinesPoints $ zipJoin testOut (toRows output)
  plotVectorsPaired [Title "Henon vs Output (2D)"] Points $ zipJoin testOut (toRows output)
  plotVectorsStyle [Title "Internal States (10,50,90)"] LinesPoints $ map (\x -> fromList [x @> 10, x@>50,x@>90]) $ toRows states
  plotMatrixEntries [Title "Trained Output Weights"] Points $ outputWeights reservoir
  return (reservoir,states,output)
  
main = mainPlot >> return ()

configs = [config{inputMatrixRange=r,outputFeedbackRange=r,internalSpectRadius=s,internalConnectivity=c} | r <- ranges,s<-radius,c<-conn]
  where
    ranges = [Continuous (-x/10,x/10) | x <- [5..10]] ++ [Continuous (-x/100,x/100) | x <- [5 .. 10]]
    radius = [x/100 | x <- [80 .. 100]]
    conn = [(x*2)/100 | x <- [5 .. 10]]
    

hMaker = do
  reservoirs <- mapM (\c -> makeReservoir c) configs
  let
    networks = reservoirs
    rregressions = [networkTrainerRRegression (tradeoff/10) | tradeoff <- [1 .. 8] :: [Double]]
  let
    result = networksProfiler runWrapper networks normalizedRootMeanSquareErrorSigma1 rregressions dummy hVals 100 500 100
  return $ result
  where
    runWrapper :: Reservoir Double -> RunReservoirM Double Noiseless (Reservoir Double,Double) -> (RunningState Noiseless Double,(Reservoir Double,Double))
    runWrapper = runReservoirNoiseless

    
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

