{-# LANGUAGE FlexibleInstances, DatatypeContexts #-}
module Reservoir.Reservoir where

import           Data.Complex                 (magnitude)
import           Data.List.Utils              (contains)
import           Data.Packed.Matrix
import           Data.Packed.Vector
import           Data.Random.Extras           (shuffle)
import           Data.Random.Source.IO
import           Data.RVar                    (sampleRVar)
import           Numeric.LinearAlgebra.Algorithms (eigenvalues)
import           Numeric.LinearAlgebra.LAPACK (eigOnlyS)
import           Numeric.LinearAlgebra.Util   (zeros)
import           System.Random                (randomIO)
import           Foreign.Storable             (Storable)

-- | Matrix that contains the connection weights among states in the neural network
type WeightMatrix a = Matrix a
type ReservoirState a = Vector a
type InputVector a = Vector a
type OutputState a = Vector a

data Reservoir a = Reservoir {
     internalState :: ReservoirState a,
     outputState :: ReservoirState a,
     inputWeights :: Maybe (WeightMatrix a),
     internalWeights :: WeightMatrix a,
     outputWeights :: WeightMatrix a,
     outputFeedbackWeights :: WeightMatrix a,
     networkFunctions :: (ReservoirState a->ReservoirState a,ReservoirState a->ReservoirState a)
}

-- | Represents the values that randomly generated can contain
data Ord a => ReservoirRange a b = Discrete (Vector b) | Continuous (a,a)

-- | The configuration parameters used to generate a ESN
data ReservoirConfiguration = ReservoirConfiguration {  
  -- | Dimension of the input vectors of the network
  inputSize :: Int, 
  -- | Range of values of the entries of the input weight matrix
  inputMatrixRange :: ReservoirRange Double Double,
  -- | Function used to fit the internal state into a particular range
  inputFunction :: (Vector Double -> Vector Double),
  -- | Number of internal nodes of the network
  internalSize :: Int,
  -- | Range of values of the internal weight matrix
  internalMatrixRange :: ReservoirRange Double Double,
  -- | The size of the expectral radius the internal matrix should have
  internalSpectRadius :: Double,
  -- | The connectivity (% of non-zero entries) of the internal matrix
  internalConnectivity :: Double,
  -- | The dimension of the network outputs
  outputSize :: Int,
  -- | The percentage of non-zero entries in the output feedback matrix
  outputFeedbackConnectivity :: Double,
  -- | The range of values in the output feedback Matrix
  outputFeedbackRange :: ReservoirRange Double Double,
  -- | The function that is applied to the result before writing it to the output channel
  outputFunction :: (Vector Double -> Vector Double)
  }

     
instance Show (Reservoir Double) where
  show (Reservoir s oState inWM intWM outWM ofbWM _) = "Reservoir " ++ (show s) ++ " " ++ (show oState) ++ " " ++ (show inWM) ++ " " ++ (show intWM) ++ " " ++ (show outWM) ++ " " ++ (show ofbWM)

-- | Generates a default configuration for the reservoir
defaultConfig inputs intNodes outputs = ReservoirConfiguration {
  inputSize = inputs,
  inputMatrixRange = Continuous (-1,1),
  inputFunction = mapVector tanh,
  internalSize = intNodes,
  internalMatrixRange = Continuous (-1,1),
  internalSpectRadius = 0.9,
  internalConnectivity = 0.1,
  outputSize = outputs,
  outputFeedbackRange = Continuous (-1,1),
  outputFeedbackConnectivity = 1,
  outputFunction = id
  }

-- | Generate a random matrix of size n x m. The entries
-- are between 0 and 1
rand :: Int -> Int -> IO (Matrix Double)
rand n m = mapMatrixWithIndexM (\_ _-> randomIO) $ zeros n m

randMatrix :: Int -> Int -> Double -> ReservoirRange Double Double -> IO (Matrix Double)
randMatrix m n conn (Discrete vals) = do
  let
    rand = shuffle $ toList vals
  mapMatrixWithIndexM (\_ _ -> sampleRVar rand >>= return . head) $ zeros m n
randMatrix m n conn (Continuous (min,max))  
  | max > min = do
    matrix' <- sprand m n conn
    let
      range = max - min
      matrix = mapMatrix (\x -> if x==0 then 0 else (range * x)+min) matrix'
    return matrix
  | otherwise = undefined -- Throw an exception
                
spectralRadius :: Matrix Double -> Double 
spectralRadius matrix = maximum $ map (abs) $ toList $ eigOnlyS matrix

setSpectralRadius :: Double -> Matrix Double -> Matrix Double
setSpectralRadius radius matrix = mapMatrix (\x -> if x == 0 then 0 else x * scaleFactor) matrix
  where 
    spectRadius = spectralRadius matrix
    scaleFactor = radius / spectRadius
    
-- | Generate a random sparse matrix with a density of
-- conn of Non-Zero elements.
sprand :: Int -> Int -> Double -> IO (Matrix Double)
sprand m n conn
  | conn >= 1 = rand m n
  | conn <= 0 = return $ zeros m n
  | otherwise = do
    let
      nnZero = ceiling ((toRational n)*(toRational m)*(toRational conn)) :: Int
      entries = shuffle [(i,j) | i<-[0..m-1],j<-[0..n-1]]
    nnZeroPos <- sampleRVar $ entries >>= \r -> return $ take nnZero r
    let
      sparseRand :: (Double,Double) -> Double -> IO Double
      sparseRand (i,j) _ = do
        case contains [(floor i,floor j)] nnZeroPos of
          True -> randomIO
          _ -> return 0
    mapMatrixWithIndexM sparseRand $ zeros m n

-- | Generate a untrained random reservoir of neural networks of
-- with the given ammount of inputs, outputs and internal units
makeReservoir :: ReservoirConfiguration -> IO (Reservoir Double)
makeReservoir conf = do
  intWM <- randMatrix units units conn intRange >>= return . (setSpectralRadius radius)
  inWM <- inputMatrix 
  ofbWM <- randMatrix outputs units ofbConn ofbRange
  let
    state = buildVector units (\_->0)
    oState = buildVector outputs (\_->0)
    outWM = zeros units outputs
  return $ Reservoir state oState inWM intWM outWM ofbWM ioFunctions
  where
    inputs = inputSize conf
    outputs = outputSize conf
    units = internalSize conf
    conn = internalConnectivity conf
    radius = internalSpectRadius conf
    intRange = internalMatrixRange conf
    inRange = inputMatrixRange conf
    ioFunctions = (inputFunction conf,outputFunction conf)
    ofbRange = outputFeedbackRange conf
    ofbConn = outputFeedbackConnectivity conf
    inputMatrix
      | inputs > 0 = randMatrix inputs units 1 inRange  >>= return . return
      | otherwise = return $ Nothing

reservoirDim :: Foreign.Storable.Storable a => Reservoir a -> Int
reservoirDim = dim . internalState

reservoirInDim reservoir = case inputWeights reservoir of
  Just inWM -> dim . head . toRows $ inWM
  Nothing -> 0

reservoirOutDim :: Foreign.Storable.Storable a => Reservoir a -> Int
reservoirOutDim = dim . outputState

updateReservoirState :: ReservoirState a -> ReservoirState a -> Reservoir a -> Reservoir a
updateReservoirState newState newOut (Reservoir _ _ inWM intWM outWM ofbWM funs) = Reservoir newState newOut inWM intWM outWM ofbWM funs

updateWeightMatrix reservoir matrix = reservoir {internalWeights = matrix}

reservoirSpectralRadius reservoir = magnitude $ foldVector (\a b -> if (magnitude  a) > (magnitude  b) then a else b) 0 $ eigenvalues $ (internalWeights reservoir)