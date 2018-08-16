(ns clojure-mxnet-lr-example.core
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.context :as context]))

;; MXNet Linear Regression Example

;; Generate data and labels with the form
;; z = a * x + b * y + c
;; a = 2.0
;; b = 1.0
;; c = 0.0 (no bias)
(def num-inputs 2) ;; x and y  <- the "data"
(def num-outputs 1) ;; z <- the "label"

(defn rand-range [n min max]
  (repeatedly n #(+ min (rand (- max min)))))

(defn linear-fn [x y]
  (+ (* x 2.0) y))

;; Describe the network
(defn lr-network []
  (as-> (sym/variable "data") data ;; placeholder for input data
    ;; I find this a little confusing since I would tend to think of
    ;; this as the output layer instead of a hidden layer.
    (sym/fully-connected "fc" {:data data :num-hidden num-outputs})
    ;; The actual output layer is what computes the training loss.
    ;; This particular type of output layer uses an L2 loss.
    (sym/linear-regression-output "lr" {:data data})))

;; Using the Module API to initialize the model, describe the expected
;; shapes of the training data and labels, initialize the optimizer,
;; and specify the devices (CPU or GPU) to use.
(defn init-module
  [nn & {:keys [batch-size learning-rate momentum]
         :or {learning-rate 0.1 momentum 0.9}}]
  (-> (m/module nn ["data"] ["lr_label"] [(context/cpu 0)])
      (m/bind {:data-shapes [{:name "data" :shape [batch-size num-inputs]}]
               :label-shapes [{:name "lr_label" :shape [batch-size num-outputs]}]})
      (m/init-params)
      (m/init-optimizer {:optimizer (optimizer/adam
                                     {:learning-rate learning-rate
                                      :momentum momentum})})))

(defn generate-batch [batch-size]
  (let [xs (rand-range batch-size -3 3)
        ys (rand-range batch-size -3 3)
        data (interleave xs ys) ;; random (x, y) data, flattened
        labels (map linear-fn xs ys)] ;; z
    ;; return data in ndarrays
    ;; The data and labels are provided in 1d form with a
    ;; shape vector that gives the number of rows, cols, etc.
    [(ndarray/array data [batch-size num-inputs])
     (ndarray/array labels [batch-size num-outputs])]))


(defn example-1 []
  (let [train-batch-size 100
        num-epoch 20
        nn (lr-network)
        mod (init-module nn :batch-size train-batch-size)
        [train-data train-label] (generate-batch train-batch-size)
        ;; Typically we package data in these ndarray iterators to pass
        ;; to the module API functions.
        train-iter (mx-io/ndarray-iter [train-data] {:label [train-label]
                                                     :label-name "lr_label"
                                                     :data-batch-size train-batch-size})
        ;; Using high level API here
        mod (m/fit mod {:train-data train-iter :num-epoch num-epoch})

        ;; Generate a smaller batch of test data to evaluate our fit model.
        test-batch-size 10
        [test-data test-label] (generate-batch test-batch-size)
        test-iter (mx-io/ndarray-iter [test-data] {:label [test-label]
                                                   :label-name "lr_label"
                                                   :data-batch-size test-batch-size})
        ;; Can use `score` to evaluate the test data with your chose metric
        [metric score] (m/score mod {:eval-data test-iter
                                     :eval-metric (eval-metric/mse)})
        ;; Here we use `predict` to get the actual results
        results (m/predict mod {:eval-data test-iter})
        ;; Getting the test data from the ndarrays so we can compare
        ;; the known label to what the model outputs.
        input-data (partition num-inputs (ndarray/->vec test-data))
        input-labels (ndarray/->vec test-label)
        ;; Grab the model predictions.
        output-labels (-> results first ndarray/->vec)
        ;; Grab the model parameters (i.e. a, b, and c)
        model-params (first (m/params mod))
        weights (ndarray/->vec (model-params "fc_weight")) ;; a and b
        bias (first (ndarray/->vec (model-params "fc_bias")))] ;; c
    (println (str "Test score: " metric " -> " score))
    (println "Sample results:")
    (doseq [[x y z] (take 4 (map vector input-data input-labels output-labels))]
      (println "Data:" x  "Label:" y "Prediction:" z))
    (println "Actual weights:" [2.0 1.0] "Model weights:" weights)
    (println "Actual bias:" 0.0 "Model bias:" bias)))

(defn example-2 []
  (let [train-batch-size 100
        num-epoch 20
        nn (lr-network)
        mod (init-module nn :batch-size train-batch-size)
        [train-data train-label] (generate-batch train-batch-size)
        train-iter (mx-io/ndarray-iter [train-data] {:label [train-label]
                                                     :label-name "lr_label"
                                                     :data-batch-size train-batch-size})
        ;; Train model as before.
        mod (m/fit mod {:train-data train-iter :num-epoch num-epoch})
        test-batch-size 5
        [test-data test-label] (generate-batch test-batch-size)
        test-iter (mx-io/ndarray-iter [test-data] {:label [test-label]
                                                   :label-name "lr_label"
                                                   :data-batch-size test-batch-size})
        ;; Same predictions as before.
        results (m/predict mod {:eval-data test-iter})
        output-labels (-> results first ndarray/->vec)
        ;; Use lower level API to get same predictions
        _ (mx-io/reset test-iter) ;; don't forget to rest the iterator
        forward-labels (-> mod
                           (m/forward (mx-io/next test-iter))
                           (m/outputs)
                           ffirst
                           ndarray/->vec)]
    ;; results should be identical
    (println "Compare results:")
    (doseq [[x y] (map vector output-labels forward-labels)]
      (println "Predict label:" x "Forward label:" y))))

;; Can use the lower level API for training the model as well
(defn my-fit [mod {:keys [train-data num-epoch]}]
  (mx-io/reset train-data)
  ;; perform multiple forward and backward passes
  (doseq [i (range num-epoch)]
    (-> mod
        (m/forward (mx-io/next train-data))
        (m/backward)
        (m/update))
    (mx-io/reset train-data))
  mod)

(defn example-3 []
  (let [train-batch-size 100
        num-epoch 20
        nn (lr-network)
        mod (init-module nn :batch-size train-batch-size)
        mod1 (init-module nn :batch-size train-batch-size)
        [train-data train-label] (generate-batch train-batch-size)
        train-iter (mx-io/ndarray-iter [train-data] {:label [train-label]
                                                     :label-name "lr_label"
                                                     :data-batch-size train-batch-size})
        ;; High level training
        mod (m/fit mod {:train-data train-iter :num-epoch num-epoch})
        ;; Lower level training
        mod1 (my-fit mod1 {:train-data train-iter :num-epoch num-epoch})

        test-batch-size 5
        [test-data test-label] (generate-batch test-batch-size)
        test-iter (mx-io/ndarray-iter [test-data] {:label [test-label]
                                                   :label-name "lr_label"
                                                   :data-batch-size test-batch-size})
        ;; Results of high level training
        results (m/predict mod {:eval-data test-iter})
        _ (mx-io/reset test-iter)
        ;; Results of low level training
        results1 (m/predict mod1 {:eval-data test-iter})
        output-labels (-> results first ndarray/->vec)
        output-labels1 (-> results1 first ndarray/->vec)]
    ;; Should be similar, small descrepancies due to differences in initialization, etc.
    (println "Compare results:")
    (doseq [[x y] (map vector output-labels output-labels1)]
      (println "fit label:" x "myfit label:" y))))
