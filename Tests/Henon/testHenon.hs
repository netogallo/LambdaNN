import Reservoir.Learning
import Reservoir.Reservoir
import Data.Packed.Vector

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
inVals = [fromList [x] | x <- [1 .. 500]]

hVals = map (\x -> mapVector (henonMap 1 1 1.4 0.3) x) inVals

initial = take 100 dummy
initialOut = take 100 hVals
trainIn = take 300 $ drop 100 dummy
trainOut = take 300 $ drop 100 hVals
testIn = take 100 $ drop 400 dummy
testOut = take 100 $ drop 400 hVals
dummy = repeat $ fromList [0]
funs = (mapVector tanh,id)

trainHenon = do
  _ <- runNetworkTeacherForced initial initialOut
  -- networkTrainerPInv trainIn trainOut
  networkTrainerRRegression trainIn trainOut
  
hLearner = do
  reservoir <- makeReservoir 0 1 50 0.2 funs
  return $ runReservoirNoiseless reservoir trainHenon

hTester = do
  (_,reservoir) <- hLearner
  return $ runReservoirNoiseless reservoir $ runNetworkCollected testIn

    
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

