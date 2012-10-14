{-# LANGUAGE FlexibleInstances #-}
module Reservoir.Reservoir where

import           Data.List.Utils              (contains)
import           Data.Packed.Matrix
import           Data.Packed.Vector
import           Data.Random.Extras           (shuffle)
import           Data.Random.Source.IO
import           Data.RVar                    (sampleRVar)
import           Numeric.LinearAlgebra.LAPACK (eigOnlyS)
import           Numeric.LinearAlgebra.Util   (zeros)
import           System.Random                (randomIO)

-- | Matrix that contains the connection weights among states in the neural network
type WeightMatrix a = Matrix a
type ReservoirState a = Vector a
type InputVector a = Vector a
type OutputState a = Vector a

data Reservoir a = Reservoir (ReservoirState a) (ReservoirState a) (WeightMatrix a) (WeightMatrix a) (WeightMatrix a) (WeightMatrix a)

instance Show (Reservoir Double) where
  show (Reservoir s oState inWM intWM outWM ofbWM) = "Reservoir " ++ (show s) ++ " " ++ (show oState) ++ " " ++ (show inWM) ++ " " ++ (show intWM) ++ " " ++ (show outWM) ++ " " ++ (show ofbWM)

-- | Generate a random matrix of size n x m. The entries
-- are between 0 and 1
rand :: Int -> Int -> IO (Matrix Double)
rand n m = mapMatrixWithIndexM (\_ _-> randomIO) $ zeros n m

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

-- | Generate a Weight matrix for the Reservoir's internal units. When generating
-- random matrixes, sometimes we get all 0 eigenvalues. This is not useful and must
-- and the matrix must be generated again. This function handles this
makeIntWeightMatrix :: Int -> Double -> IO (Matrix Double)
makeIntWeightMatrix units conn = do
  rawIntWM <- (sprand units units conn) >>= return . (mapMatrix (\n -> if n == 0 then 0 else n - 0.5))
  let
    maxEigVal = maximum $ map (abs) $ toList $ eigOnlyS $ rawIntWM
  case maxEigVal of
    0 -> makeIntWeightMatrix units conn
    _ -> return $ rawIntWM / (fromLists [[maxEigVal]])

-- | Generate a untrained random reservoir of neural networks of
-- with the given ammount of inputs, outputs and internal units
makeReservoir :: Int -> Int -> Int -> Double -> IO (Reservoir Double)
makeReservoir inputs outputs units conn = do
  intWM <- makeIntWeightMatrix units conn
  inWM <- rand units inputs
  ofbWM <- rand units outputs
  let
    state = buildVector units (\_->0)
    oState = buildVector outputs (\_->0)
    outWM = zeros outputs (units+inputs)
  return $ Reservoir state oState inWM intWM outWM ofbWM


reservoirDim (Reservoir state _ _ _ _ _) = dim state

reservoirInDim (Reservoir _ _ inWM _ _ _) = dim . head . toRows $ inWM

reservoirOutDim (Reservoir _ oState _ _ _ _) = dim oState
