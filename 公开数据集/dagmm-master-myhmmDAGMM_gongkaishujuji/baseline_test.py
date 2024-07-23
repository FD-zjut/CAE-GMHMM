from adbench.run import RunPipeline

'''
Params:
suffix: file name suffix;

parallel: running either 'unsupervise', 'semi-supervise', or 'supervise' (AD) algorithms,
corresponding to the Angle I: Availability of Ground Truth Labels (Supervision);

realistic_synthetic_mode: testing on 'local', 'global', 'dependency', and 'cluster' anomalies, 
corresponding to the Angle II: Types of Anomalies;

noise type: evaluating algorithms on 'duplicated_anomalies', 'irrelevant_features' and 'label_contamination',
corresponding to the Angle III: Model Robustness with Noisy and Corrupted Data.
'''

# return the results including [params, model_name, metrics, time_fit, time_inference]
# besides, results will be automatically saved in the dataframe and ouputted as csv file in adbench/result folder
# pipeline = RunPipeline(suffix='ADBench', parallel='semi-supervise', realistic_synthetic_mode=None, noise_type=None)
# results = pipeline.run()

pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise', realistic_synthetic_mode='cluster', noise_type=None)
results = pipeline.run()

# pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode=None, noise_type='irrelevant_features')
# results = pipeline.run()