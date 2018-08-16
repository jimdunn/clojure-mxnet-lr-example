(ns clojure-mxnet-lr-example.examples
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

;; ndarray - multi-dim arrays - annoyingly referred to as tensors
(defn ndarray-ex []
  (let [data [0 1 2 3 4 5]
        shape [1 6]
        shape1 [6 1]
        array-0 (ndarray/array data shape)
        array-1 (ndarray/ones shape)
        ;array-1 (ndarray/ones shape1)
        array-2 (ndarray/+ array-0 array-1)]
    (ndarray/->vec array-2)))

(defn symbol-ex []
  (let [x (sym/variable "x")
        y (sym/variable "y")
        ;z (sym/+ x y)
        z (sym/+ (sym/* x 2.0) y)]
    (-> z
        (sym/bind {"x" (ndarray/ones [3])
                   "y" (ndarray/ones [3])})
        executor/forward
        executor/outputs
        first
        ndarray/->vec)))

;; z = a * x + b * y + c
;; a = 2, b = 1, c = 0 (no bias)
(defn linear-fn []
  (let [x (sym/variable "x")
        y (sym/variable "y")]
    (sym/+ (sym/* x 2.0) y)))

;; generating random data using mxnet for some reason
(defn generate-data [batch-size]
  (let [data* (random/uniform -3 3 [2 batch-size])
        label* (-> (linear-fn)
                   (sym/bind {"x" (ndarray/slice data* 0)
                              "y" (ndarray/slice data* 1)})
                   executor/forward
                   executor/outputs
                   first)
        data (ndarray/transpose data*)
        label (ndarray/transpose label*)]
    (mx-io/ndarray-iter [data] {:label [label]
                                :label-name "lr_label"
                                :data-batch-size batch-size})))

;; linear regression
(defn lr-network [num-outputs]
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc" {:data data :num-hidden num-outputs})
    (sym/linear-regression-output "lr" {:data data})))

;; module api
(defn init-module
  [nn num-inputs num-outputs & {:keys [batch-size learning-rate momentum]
                                :or {learning-rate 0.1 momentum 0.9}}]
  (-> (m/module nn ["data"] ["lr_label"] [(context/cpu 0)])
      (m/bind {:data-shapes [{:name "data" :shape [batch-size num-inputs]}]
               :label-shapes [{:name "lr_label" :shape [batch-size num-outputs]}]})
      (m/init-params)
      (m/init-optimizer {:optimizer (optimizer/adam
                                     {:learning-rate learning-rate
                                      :momentum momentum})})))

(defn module-ex []
  (let [num-inputs 2
        num-outputs 1
        train-batch-size 100
        train-iter (generate-data train-batch-size)
        num-epoch 20
        nn (lr-network num-outputs)
        mod (init-module nn num-inputs num-outputs :batch-size train-batch-size)]
    (m/fit mod {:train-data train-iter :num-epoch num-epoch})))

(defn model-weights [mod]
  (let [model-params (first (m/params mod))
        weights (ndarray/->vec (model-params "fc_weight"))
        bias (first (ndarray/->vec (model-params "fc_bias")))]
    (println "Actual weights:" [2.0 1.0] "Model weights:" weights)
    (println "Actual bias:" 0.0 "Model bias:" bias)))

(defn predict-ex [mod]
  (let [test-batch-size 10
        test-iter (generate-data test-batch-size)
        results (m/predict mod {:eval-data test-iter})
        output-labels (-> results first ndarray/->vec)
        test-label (-> test-iter mx-io/iter-label first ndarray/->vec)]
    (doseq [[model-output expected-output] (map vector output-labels test-label)]
      (println "Actual:" model-output "Expected:" expected-output))))
