import Reservoir.Learning
import System.IO
import Data.Packed.Vector
import Reservoir.Reservoir

loadTimeSeries h = do
  series <- hGetContents h >>= return . lines
  s <- return $ (map read series :: [Double])
  return s


trainTest h = do
  series <- loadTimeSeries h
  let
    training = zipWith (\x y->(fromList [x],fromList [y])) [1..1500] $ take 1500 series
    testing = [fromList [x] | x <- [1501 .. 2000]]
    realData = map (fromList.return) $ drop 1500 series
  untrainedNN <- makeReservoir 1 1 100 0.1
  trainedNN <- runNetworkTrainingIO 0 200 (id,id) untrainedNN training
  (seq,nn) <- runNetworkIO 0 (id,id) trainedNN testing
  return (nn,seq,realData)

main = do  
  h <- openFile "./henon.dat" ReadMode
  (nn,seq,realData) <- trainTest h
  out <- openFile "./out.dat" WriteMode
  let
    packedData = zipWith (\x y -> (head $ toList x,head $ toList y)) seq realData
  mapM (\(x,y) -> hPutStrLn $ (show x) ++ " " ++ (show y)) packedData
  hClose h
  hClose out