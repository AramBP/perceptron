open Owl


let activation_func h beta = 1. /. (1. +. exp(-1. *. beta *. h))
let append_bias_col a: Mat.mat = 
  let n_rows = Mat.row_num a in
  let col = Mat.create n_rows 1 (-1.) in
  Mat.concat_horizontal a col

let hidden_layers_activations (inputs: Mat.mat) (weights1: Mat.mat) (beta: float): Mat.mat =
  let hidden =  Mat.(inputs *@ weights1) in
  let hidden' = append_bias_col (Mat.map (fun x -> activation_func x beta) hidden) in
  hidden'

let output_layer_activations (hidden: Mat.mat) (weights2: Mat.mat) (beta: float): Mat.mat =
  let outputs = Mat.(hidden *@ weights2) in
  let outputs' = Mat.map (fun x -> activation_func x beta) outputs in
  outputs'

let train (inputs: Mat.mat) (targets: Mat.mat) (eta: float) (niterations: int) (nhidden: int) =  
  let nIn = Mat.col_num inputs in
  let nOut = Mat.row_num targets in
  let inputs' = append_bias_col inputs in
  
  (*initialize weight matrices to small random values*)
  let weights1 = Mat.map (( *. ) (2. /. sqrt((float) nIn))) (Mat.map ((-.) 0.5) (Mat.uniform (nIn+1) nhidden)) in
  let weights2 = Mat.map (( *. ) (2. /. sqrt((float) nIn))) (Mat.map ((-.) 0.5) (Mat.uniform (nhidden+1) nOut)) in
  
  let rec loop n w1 w2 = 
    if n <= 0 then ()
    else
      (*forwards fase*)
      let hidden = hidden_layers_activations inputs' w1 1. in
      let outputs = output_layer_activations hidden w2 1. in
      Mat.print hidden;
      Mat.print outputs;

      let square x = x *. x in
      let error =  0.5 *. Mat.(sum' (map square (outputs - targets))) in
      Printf.printf "Iteration :%d Error: %0.10f\n" (niterations - n) error;
      
      (*backwards fase*)
      
      (*compute error at the output*)
      let deltao = Mat.((outputs - targets) *@ outputs *@ (ones (row_num outputs) (col_num outputs) - outputs)) in
      Mat.print deltao;
      (*update output layer weights*)
      let updatew2 = Mat.(weights2 - (map (fun x -> x *. eta) ((transpose hidden) *@ deltao))) in
      Mat.print updatew2;
      (* compute error at the hidden layers *)
      let deltah = Mat.(hidden *@ (ones (row_num hidden) (col_num hidden) - hidden) *@ (deltao *@ (transpose weights2))) in
      Mat.print deltah;
      (*update hidden layer weights*)
      let deltah_sliced = Mat.get_slice [[]; [0; (Mat.col_num deltah) - 2]] deltah in (*select every column except the last one from deltah*)
      let updatew1 = Mat.(weights1 - (map (fun x -> x *. eta) ((transpose inputs') *@ deltah_sliced ))) in
      
      loop (n-1) updatew1 updatew2
  in

  loop niterations weights1 weights2;
  Printf.printf ""

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
  train and_input and_targets 0.25 1001 nhidden