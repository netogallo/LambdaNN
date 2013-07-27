-- | LearningUtils
-- Contains functions used to train and test multiple networks on a particular task. This functions should be used to automate
-- the tuning of the neural networks.
module Reservoir.LearningUtils where

import Reservoir.Learning
import Reservoir.Reservoir (Reservoir)
import Control.Parallel.Strategies (parMap,rpar,rdeepseq)
import Data.Packed.Matrix (toRows,Matrix,toLists,Element,cols)
import Reservoir.Reservoir (ReservoirState)
import Data.Packed (Vector,join,fromList,toList)
import Data.Vector.Storable ()
import Data.Packed.Vector(zipVector,foldVector,dim,(@>),buildVector)
import Graphics.Gnuplot.Simple
import Graphics.Gnuplot.Value.Tuple (C)
import Foreign.Storable  (Storable)
import Numeric.LinearAlgebra.Util (norm)

autoSelectStates :: Int -> Matrix Double -> [Vector Double]
autoSelectStates n vs = selectStates sStates vs
  where
    step = cols vs `div` n
    sStates = [x*step | x <- [0..(n-1)]]

mse :: [Vector Double] -> [Vector Double] -> Double
mse xs ys = sum $ map (\(x,y) -> norm $ x - y) $ zip xs ys                              

selectStates :: (Element a,Storable a) => [Int] -> Matrix a -> [Vector a]
selectStates entries m = map (makeVector) $ toRows m
  where
    makeVector v = fromList $ map (\e -> v @> e) entries

dimRange :: Storable a => Vector a -> [Int]
dimRange vector = [0 .. ((dim vector) -1)]

zipJoin :: Storable a => [Vector a] -> [Vector a] -> [Vector a]
zipJoin vec1 vec2 = map (\(x,y) -> join [x,y]) $ zip vec1 vec2

--plotPairs :: (C a,Storable a,Storable (a,a)) => [Attribute] -> PlotType -> [(Vector a,Vector a)] -> IO ()
plotPairs attr pType vectors = plotListStyle attr (PlotStyle pType $ DefaultStyle 1) values
  where    
    values = foldr (\(v1,v2) accm -> (zip (toList v1) (toList v2)) ++ accm) [] vectors

plotVectorsPaired :: (C a,Storable a) => [Attribute] -> PlotType -> [Vector a] -> IO ()
plotVectorsPaired attr pType vectors = plotListsStyle attr values
  where
    pairIndexes i = (\(_,res) -> res) . foldl (\(last,res) vector -> (vector,(last @> i,vector @> i):res)) ((head vectors),[])
    values = [(PlotStyle{plotType=pType,lineSpec=DefaultStyle $ i + 1},pairIndexes i vectors) | i <- (dimRange $ head vectors)] 

plotVectorsStyle :: (C a, Storable a) => [Attribute] -> PlotType -> [Vector a] -> IO ()
plotVectorsStyle style pType vectors = plotListsStyle style values
  where
    takeIndex i = map (@>i) vectors
    values = [ (PlotStyle{plotType=pType,lineSpec=DefaultStyle $ i + 1},takeIndex i) | i <- [0 .. ((dim $ head vectors)-1)]]
    
plotMatrixEntries :: (C a, Storable a,Element a) => [Attribute] -> PlotType -> Matrix a -> IO ()
plotMatrixEntries style pType matrix = plotListsStyle style values
  where
    values = [(PlotStyle{plotType=pType,lineSpec=DefaultStyle 1}, concat $ toLists matrix)]
    
toDouble :: Int -> Double
toDouble = fromInteger . toInteger

normalizedRootMeanSquareError :: [Vector Double] -> [Vector Double] -> [Vector Double] -> Double
normalizedRootMeanSquareError sigmasSq teach output = (foldl nrmse 0 values)/(toDouble $ length values)
  where
    values = zip3 teach output sigmasSq
    nrmse error (teach',output',sigmaSq) = let
      in
       error + (foldl (\e i -> e + nrmse' (sigmaSq @> i,teach' @> i,output' @> i)) 0 [0 .. ((dim output') -1)])/(toDouble $ dim output')
    nrmse' (sigmaSq,teach',output') = sqrt (((teach'-output')**2)/sigmaSq)
    
normalizedRootMeanSquareErrorSigma1 :: [Vector Double] -> [Vector Double] -> Double
normalizedRootMeanSquareErrorSigma1 teach output = normalizedRootMeanSquareError sigmasSq teach output
  where
    sigmasSq = repeat $ buildVector (dim $ head teach) $ (\_ -> 1)

profileNetworkTecherForced error [] _ inputs outputs = do  
  (intStates,outStates) <- runNetworkCollectedTeacherForced inputs outputs
  return (error (toRows outStates) outputs)  
profileNetworkTecherForced error initialIn initialOut inputs outputs = do
  _ <- runNetworkTeacherForced initialIn initialOut
  profileNetworkTecherForced error [] [] inputs outputs
  
networksProfiler :: (Reservoir Double -> RunReservoirM Double rand (Reservoir Double, Double) -> (RunningState t Double,(Reservoir Double, Double)))
                    -> [Reservoir Double]
                    -> ([Vector Double] -> [Vector Double] -> Double)
                    -> [[Vector Double]
                        -> [Vector Double]
                        -> RunReservoirM Double rand (Reservoir Double)]
                    -> [Vector Double]
                    -> [Vector Double]
                    -> Int
                    -> Int
                    -> Int
                    -> (Reservoir Double, Double)
networksProfiler networkRunner networks errorMeasure trainingFunctions inputs outputs initialSize trainingSize testSize = let
  networkProfiler' = networkProfiler errorMeasure inputs outputs initialSize trainingSize testSize
  profileFun (network,trainer) = networkRunner network $ networkProfiler' trainer
  ((_,r):results) = parMap rpar profileFun [(n,t) | n <- networks, t<-trainingFunctions]
  in
   foldl (\(n,e) (_,(n',e')) -> if e < e' then (n,e) else (n',e')) r results

networkProfiler :: ([Vector Double] -> [Vector Double] -> Double)
                   -> [Vector Double]               
                   -> [ReservoirState Double]
                   -> Int
                   -> Int
                   -> Int
                   -> ([Vector Double] -> [ReservoirState Double] -> RunReservoirM Double rand (Reservoir Double))
                   -> RunReservoirM Double rand (Reservoir Double, Double)
networkProfiler errorMeasure inputs outputs initialSize trainingSize testSize trainingFunction = do
  _ <- runNetworkTeacherForced initial initialOut
  reservoir <- trainingFunction trainIn trainOut
  (_,results) <- runNetworkCollected testIn
  let
    cmpList = zip testOut $ toRows results
    performance = errorMeasure testOut $ toRows results
  return (reservoir,performance)
  where
    initial = take initialSize inputs
    initialOut = take initialSize outputs
    trainIn = take trainingSize $ drop initialSize inputs
    trainOut = take trainingSize $ drop initialSize outputs
    testIn = take testSize $ drop (initialSize+trainingSize) inputs
    testOut = take testSize $ drop (initialSize+trainingSize) outputs
  