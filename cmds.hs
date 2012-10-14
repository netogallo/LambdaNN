import System.IO.Unsafe
let x = unsafePerformIO $ makeReservoir 2 3 4 0.2

let ofbWM = unsafePerformIO $ rand 4 3

let noise:_ = toColumns $ (*) 0.1 $ (unsafePerformIO $ rand 4 1) - 0.5

import System.IO.Unsafe
let input:_ = toColumns $ (unsafePerformIO $ rand 2 1) - 0.5
let teach:_ = toColumns $ (unsafePerformIO $ rand 3 1) - 0.5