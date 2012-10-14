import Reservoir.Learning
import System.IO
import Data.Packed.Vector
import Reservoir.Reservoir

loadTimeSeries h = do
  series <- hGetContents h >>= return . lines
  s <- return $ (map read series :: [Double])
  return s


trainSize = 2500
funs = (tan,atan)

trainTest h = do
  series <- loadTimeSeries h
  let
    training = zipWith (\x y->(fromList [x],fromList [y])) [0 | x <- [1..trainSize]] $ take trainSize series
    testing = [fromList [0] | x <- [(trainSize +1) .. (trainSize + 500)]]
    realData = map (fromList.return) $ drop trainSize series
  untrainedNN <- makeReservoir 1 1 200 0.1
  trainedNN <- runNetworkTrainingIO 0 500 funs untrainedNN training
  (seq,nn) <- runNetworkIO 0 funs trainedNN testing
  return (nn,seq,realData)

main = do  
  h <- openFile "./henon.dat" ReadMode
  (nn,seq,realData) <- trainTest h
  out <- openFile "./out.dat" WriteMode
  let
    packedData = zipWith (\x y -> (head $ toList x,head $ toList y)) seq realData
  mapM (\(x,y) -> hPutStrLn out $ (show x) ++ " " ++ (show y)) packedData
  hClose h
  hClose out