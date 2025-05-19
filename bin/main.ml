open Owl


let activation_func h beta = 1. /. (1. +. exp(-1. *. beta *. h))
let append_bias_col a: Mat.mat = 
  let n_rows = Mat.row_num a in
  let col = Mat.create n_rows 1 (-1.) in
  Mat.concat_horizontal a col

let fwd (inputs: Mat.mat) (weights1: Mat.mat) (weights2: Mat.mat) (beta: float)=
  let hidden =  Mat.(inputs *@ weights1) in
  let hidden' = append_bias_col (Mat.map (fun x -> activation_func x beta) hidden) in
  let outputs = Mat.(hidden' *@ weights2) in
  Mat.print hidden';
  Mat.print outputs

let train (inputs: Mat.mat) (targets: Mat.mat) (*(eta: float) (niterations: int)*) (nhidden: int) =  
  let nIn = Mat.col_num inputs in
  (*let nData = Mat.row_num inputs in*)
  let nOut = Mat.row_num targets in
  let inputs' = append_bias_col inputs in
  Mat.print inputs';
  
  (*initialize weight matrices to small random values*)
  let weights1 = Mat.map (( *. ) (2. /. sqrt((float) nIn))) (Mat.map ((-.) 0.5) (Mat.uniform (nIn+1) nhidden)) in
  let weights2 = Mat.map (( *. ) (2. /. sqrt((float) nIn))) (Mat.map ((-.) 0.5) (Mat.uniform (nhidden+1) nOut)) in
  
  fwd inputs' weights1 weights2 1.


let () =
  (* AND data*)
  let anddata_arr = [|
    [|0.0; 0.0; 0.0|];
    [|0.0; 1.0; 0.0|];
    [|1.0; 0.0; 0.0|];
    [|1.0; 1.0; 1.0|]
  |] in
  
  let anddata = Mat.of_arrays anddata_arr in
  let and_input = Owl.Mat.cols anddata [|0;1|] in
  let and_targets = Owl.Mat.col anddata 2 in
  let nhidden = 2 in

  train and_input and_targets nhidden
  