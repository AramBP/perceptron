open Owl

(*
See below for all the function signatures used in the perceptron
module type PerceptronFull = sig 
  val append_bias_col: Mat.mat -> Mat.mat
  val activation_func: float -> float
  val hidden_layers_activations: Mat.mat -> Mat.mat -> Mat.mat
  val output_layer_activations: Mat.mat -> Mat.mat -> Mat.mat
  val argmax_rows: Mat.mat -> float array
  val confmat: Mat.mat -> Mat.mat -> Mat.mat
  val train: Mat.mat -> Mat.mat -> unit
end
*)
module type Perceptron = sig 
  val train: Mat.mat -> Mat.mat -> unit
end
 
module type PerceptronConstants = sig 
  val beta: float
  val eta: float
  val niterations: int
  val nhidden: int
end

module Make (C: PerceptronConstants): Perceptron = struct 
  let activation_func h = 1. /. (1. +. exp(-1. *. C.beta *. h))

  let append_bias_col a: Mat.mat = 
    let n_rows = Mat.row_num a in
    let col = Mat.create n_rows 1 (-1.) in
    Mat.concat_horizontal a col
  
  let hidden_layers_activations (inputs: Mat.mat) (weights1: Mat.mat): Mat.mat =
    let hidden =  Mat.(inputs *@ weights1) in
    let hidden' = append_bias_col (Mat.map (fun x -> activation_func x) hidden) in
    hidden'
  
  let output_layer_activations (hidden: Mat.mat) (weights2: Mat.mat): Mat.mat =
    let outputs = Mat.(hidden *@ weights2) in
    let outputs' = Mat.map (fun x -> activation_func x) outputs in
    outputs'

  let argmax_rows (a: Mat.mat)  = 
    let nrows = Mat.row_num a in
    let ncols = Mat.col_num a in
    let output = Array.make nrows 0. in

    for i = 0 to nrows - 1 do
      let max = ref (Mat.get a i 0) in
      output.(i) <- 0.;
      for j = 0 to ncols -1 do
        let elem = ref (Mat.get a i j) in
        if !elem > !max then 
          max := !elem;
          output.(i) <- (float) j;
      done;
    done;
    output

  let confmat (outputs: Mat.mat) (targets: Mat.mat) =

    let compute_confmat predicted targets' nclasses = 
      let cm = Mat.zeros nclasses nclasses in
      for i = 0 to nclasses-1 do
        for j = 0 to nclasses-1 do
          let predicted_where = Mat.map (fun x -> if x = (float) i then 1. else 0.) predicted in
          let targets_where = Mat.map (fun x -> if x = (float) j then 1. else 0.) targets' in
          cm.Mat.%{i;j} <- Mat.(sum' ( predicted_where * targets_where));
        done;
      done;
      cm
    in

    let ncols_targets = Mat.col_num targets in
    if ncols_targets = 1 then 
      let nclasses = 2 in
      let predicted = Mat.map (fun x -> if x > 0.5 then 1. else 0.) outputs in
      compute_confmat predicted targets nclasses
    else 
      (*One of N encoding*)
      let nclasses = ncols_targets in
      let predicted = Mat.of_array (argmax_rows outputs) 1 nclasses in
      let targets' = Mat.of_array (argmax_rows targets) 1 nclasses in
      compute_confmat predicted targets' nclasses

  let train (inputs: Mat.mat) (targets: Mat.mat) =  
    let nIn = Mat.col_num inputs in
    let nOut = Mat.col_num targets in
    let inputs' = append_bias_col inputs in
    
    (*initialize weight matrices to small random values*)
    let weights1 = Mat.map (( *. ) (2. /. sqrt((float) nIn))) (Mat.map ((-.) 0.5) (Mat.uniform (nIn+1) C.nhidden)) in
    let weights2 = Mat.map (( *. ) (2. /. sqrt((float) nIn))) (Mat.map ((-.) 0.5) (Mat.uniform (C.nhidden+1) nOut)) in
    
    let rec loop n w1 w2 = 
      let hidden = hidden_layers_activations inputs' w1 in
      let outputs = output_layer_activations hidden w2 in
      if n <= 0 then begin
        (*generate and display confusion matrix*)
        let cm = confmat outputs targets in
        let percentage_correct = ((Mat.trace cm) /. (Mat.sum' cm)) *. 100. in
        Printf.printf "\nPercentage correct: %0.2f percent\nConfusion matrix:" percentage_correct;
        Mat.print cm

      end
      else begin
        (*forwards fase*)
        let square x = x *. x in
        let error =  0.5 *. Mat.(sum' (map square (outputs - targets))) in
        Printf.printf "Iteration :%d Error: %0.10f\n" (C.niterations - n) error;
        
        (*backwards fase*)
        
        (*compute error at the output*)
        let deltao = Mat.((outputs - targets) * outputs * (ones (row_num outputs) (col_num outputs) - outputs)) in
        (*update output layer weights*)
        let updatew2 = Mat.(w2 - (map (fun x -> x *. C.eta) ((transpose hidden) *@ deltao))) in
        (* compute error at the hidden layers *)
        let deltah = Mat.(hidden * (ones (row_num hidden) (col_num hidden) - hidden) * (deltao *@ (transpose w2))) in
        (*update hidden layer weights*)
        let deltah_sliced = Mat.get_slice [[]; [0; (Mat.col_num deltah) - 2]] deltah in (*select every column except the last one from deltah*)
        let updatew1 = Mat.(w1 - (map (fun x -> x *. C.eta) ((transpose inputs') *@ deltah_sliced ))) in
        
      
        loop (n-1) updatew1 updatew2
      end
    in
    loop C.niterations weights1 weights2;
    Printf.printf ""

end

module MyPerceptron = Make (struct 
    let beta = 1.0
    let eta = 0.25
    let niterations = 1001
    let nhidden = 2
end) 

let () =
  (* AND data*)
  let anddata_arr = [|
    [|0.0; 0.0; 0.0|];
    [|0.0; 1.0; 0.0|];
    [|1.0; 0.0; 0.0|];
    [|1.0; 1.0; 1.0|]
  |] in
  
  let anddata = Mat.of_arrays anddata_arr in
  let and_input = Mat.cols anddata [|0;1|] in
  let and_targets = Mat.col anddata 2 in

  MyPerceptron.train and_input and_targets

