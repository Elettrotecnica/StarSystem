package mining;

import java.io.File;
import java.io.FilenameFilter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import conf.Configuration;

import utils.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;

public class TestSetEvaluation {
	
	final Configuration m_conf;
	
	final ExecutorService m_threadExecutor;
	
	final String m_trainFolder;
	final String m_testFolder;
	final String m_resultsFolder;
    
    public TestSetEvaluation(final Configuration conf, final ExecutorService threadExecutor) {
    	m_conf = conf;
    	if (threadExecutor != null) {
    		m_threadExecutor = threadExecutor;
    	} else {
    		m_threadExecutor = Executors.newFixedThreadPool(m_conf.m_nThreads);
    	}
    	
    	m_trainFolder = m_conf.crossValidationSelectedDatasetPath();
    	m_testFolder  = m_conf.testSetSelectedDatasetPath();
    	
    	final String[] resultsFolderTokens = {m_conf.m_baseFolder, m_conf.m_testSetFolder, m_conf.m_resultsFolder, ""};
		m_resultsFolder = Utils.join(resultsFolderTokens, m_conf.m_fileSeparator);
    }
    
    public TestSetEvaluation(Configuration conf) {
    	this(conf, null);
    }
    
    public TestSetEvaluation() {
    	this(new Configuration());
    }
	
	private String stats(
			final Evaluation eval, 
			final String title) throws Exception {
		return eval.toSummaryString(title, true)
		     + eval.toClassDetailsString();
	}
	
//	private void addClassification(
//			final Instances trainingSet, 
//			final Instances testSet, 
//			final Classifier model) throws Exception {
//		final AddClassification acl = new AddClassification();
//		acl.setClassifier(model);
//		acl.setRemoveOldClass(false);
//		acl.setOutputClassification(true);
//		acl.setInputFormat(trainingSet);
//		Utils.saveInstances(Filter.useFilter(testSet, acl), File.createTempFile("Predictions", ".arff"));
//	}
	
	private Evaluation evalOneOne(
			final Instances trainingSet, 
			final Instances testSet, 
			final Classifier model) throws Exception {
		final Evaluation eval = new Evaluation(trainingSet);
		final int nInstances = testSet.numInstances();
		for (int i = 0; i < nInstances; i++) {
			model.buildClassifier(trainingSet);
			final Instance in = testSet.instance(i);
			eval.evaluateModelOnceAndRecordPrediction(model, in);
			trainingSet.add(in);
		}
		return eval;
	}
	
	private Evaluation eval(
			final Instances trainingSet, 
			final Instances testSet, 
			final Classifier model) throws Exception {
		final Evaluation eval = new Evaluation(trainingSet);
		model.buildClassifier(trainingSet);
		eval.evaluateModel(model, testSet);
		return eval;
	}
	
	private Void exp(
			final String competition, 
			final String nFeatures, 
			final String modelName,
			final String[] modelOptions,
			final double CVPerformance, 
			final double CVStdDev,
			final double CVBaselinePerformance,
			final double CVBaselineStdDev) throws Exception {
		// Filter for every dataset file for this competition and number of features
		final String regexp = "^" + competition + ".*features" + nFeatures + ".arff$";
		final FilenameFilter fileFilter = Utils.getFileFilter(regexp);
		
		// Create instance of classfier from its specs
		final Classifier model = (Classifier) weka.core.Utils.forName(Classifier.class, modelName, modelOptions);
		// Output file name suffix
		final String outFileNameSuffix = "-Prediction-" + modelName + ".txt";
		// Get datasets folder...
		final File dir = new File(m_trainFolder);
		// ...then run experiment for every file of this competition and features (currently this will be only one file)
		for (final File trainFile : dir.listFiles(fileFilter)) {
			final String trainFileName = trainFile.getPath();
			final String testFileName  = trainFileName.replace(m_trainFolder, m_testFolder);
									
			// Read training set
			Instances trainingSet = Utils.readFile(trainFile);
			final int nTrainInstances = trainingSet.numInstances();
			
			// Read test set
			Instances testSet = Utils.readFile(testFileName);
			final int nTestInstances = testSet.numInstances();
			
			// Note: - datasets are already ordered by sort-attribute-name
			//       - datasets have already been deprived from sorting attributes 
			
			// 3 avaluators for baseline, model with retrain, model without retrain
			final Evaluation zeroREval     = eval(trainingSet, testSet, new ZeroR());
			final Evaluation noRetrainEval = eval(trainingSet, testSet, model);
			final Evaluation retrainEval   = evalOneOne(trainingSet, testSet, model);
			
			final double zeroRPerformance;
			final double noRetrainPerformance;
			final double retrainPerformance;
		    switch (m_conf.m_evaluationMeasure) {
		    	case "accuracy":  
					zeroRPerformance = zeroREval.pctCorrect();
					noRetrainPerformance = noRetrainEval.pctCorrect();
					retrainPerformance = retrainEval.pctCorrect();
		    		break;
		    	case "precision":  
					zeroRPerformance = zeroREval.weightedPrecision();
					noRetrainPerformance = noRetrainEval.weightedPrecision();
					retrainPerformance = retrainEval.weightedPrecision();
		    		break;
		    	case "recall":  
					zeroRPerformance = zeroREval.weightedRecall();
					noRetrainPerformance = noRetrainEval.weightedRecall();
					retrainPerformance = retrainEval.weightedRecall();
		    		break;
		    	case "f-measure":  
					zeroRPerformance = zeroREval.weightedFMeasure();
					noRetrainPerformance = noRetrainEval.weightedFMeasure();
					retrainPerformance = retrainEval.weightedFMeasure();
		    		break;
		    	default :  
					zeroRPerformance = zeroREval.pctCorrect();
					noRetrainPerformance = noRetrainEval.pctCorrect();
					retrainPerformance = retrainEval.pctCorrect();
	    			System.out.println("WARNING Unrecognized evaluation mesure. Using Accuracy as default.");
		    		break;
		    }
			
			// Does model without retrain outperform baseline?
			final boolean noRetrainOutperformsBaseline = 
					noRetrainPerformance > zeroRPerformance && Math.abs(noRetrainPerformance - CVPerformance) <= CVStdDev;
					
			// Does model with retrain outperform baseline?
			final boolean retrainOutperformsBaseline = 
					retrainPerformance > zeroRPerformance && Math.abs(retrainPerformance - CVPerformance) <= CVStdDev;

			// We are only keeping results that outperform baseline significantly somehow
			if (noRetrainOutperformsBaseline || retrainOutperformsBaseline) {
				// Get output file writer
				final String featureSelectionSpec = trainFile.getName().replaceAll(".arff$", "");
				final String outFileName = m_resultsFolder + m_conf.m_fileSeparator + featureSelectionSpec + outFileNameSuffix;
				final PrintWriter writer = new PrintWriter(outFileName, "UTF-8");
				
				writer.println("#################################");
				writer.println("## Conf. outperforms baseline? ##");
				writer.println("#################################");
				if (noRetrainOutperformsBaseline) {
				writer.println("## Without Retrain:        YES ##");
				} else {
				writer.println("## Without Retrain:        NO  ##");
				}
				if (retrainOutperformsBaseline) {
				writer.println("## With Retrain:           YES ##");
				} else {
				writer.println("## With Retrain:           NO  ##");
				}
				writer.println("#################################");
	
				writer.println();
				writer.println("##########  SUMMARY  ##########");
				writer.println("## Dataset           : " + competition);
				writer.println("## Training set      : " + trainFileName);
				writer.println("## Test set          : " + testFileName);
				writer.println("## N. Features       : " + nFeatures);
				writer.println("## Model             : " + modelName);
				writer.println("## Model Options     : " + Utils.join(model.getOptions(), " "));
				writer.println("## Measure           : " + m_conf.m_evaluationMeasure);
				writer.println("## CV Performance    : " + CVPerformance + " +/- " + CVStdDev);
				writer.println("## CV Baseline       : " + CVBaselinePerformance + " +/- " + CVBaselineStdDev);
				writer.println("## Train cardinality : " + nTrainInstances);
				writer.println("## Test cardinality  : " + nTestInstances);
				writer.println("## This Baseline     : " + zeroRPerformance);
				writer.println("## With Retrain      : " + retrainPerformance);
				writer.println("## Without Retrain   : " + noRetrainPerformance);
				writer.println("###############################");
				
				writer.println(stats(noRetrainEval, "\nNo retrain\nResults\n======\n"));
				writer.println(stats(retrainEval,   "\nRetrain\nResults\n======\n"));
				
				writer.close();
			}
		}
		return null;
	}
	
	private Future<Void> submitExp(
			final String competition, 
			final String nFeatures, 
			final String modelName,
			final String[] modelOptions,
			final double CVPerformance, 
			final double CVStdDev,
			final double CVBaselinePerformance,
			final double CVBaselineStdDev) throws Exception {		
		// Accodo l'esperimento nell'esecutore
		return m_threadExecutor.submit(new Callable<Void>() {
			public Void call() throws Exception {
				return exp(competition, nFeatures, modelName, modelOptions, CVPerformance, CVStdDev, CVBaselinePerformance, CVBaselineStdDev);
		    }
		});
	}
	
	private List<Future<Void>> experiment(final List<String[]> settings) throws Exception {
		final List<Future<Void>> queue = new ArrayList<Future<Void>>();
		for (final String[] setting : settings) {
			final String competition   = setting[0];
			final String nFeatures     = setting[1];
			final String modelName     = setting[2];
			final String[] options     = weka.core.Utils.splitOptions(setting[3]);
			final double CVPerformance = Double.parseDouble(setting[4]);
			final double CVStdDev      = Double.parseDouble(setting[5]);
			final double CVBaselinePerformance = Double.parseDouble(setting[6]);
			final double CVBaselineStdDev      = Double.parseDouble(setting[7]);
			queue.add(submitExp(competition, nFeatures, modelName, options, CVPerformance, CVStdDev, CVBaselinePerformance, CVBaselineStdDev));
		}
		return queue;
	}
	
	public void incrementalExperiment(final List<Future<List<String[]>>> results) throws Exception {
		final List<Future<Void>> queue = new ArrayList<Future<Void>>();
		for (final Future<List<String[]>> r : results) {
			final List<String[]> settings = r.get();
			queue.addAll(experiment(settings));
		}; for (final Future<Void> done : queue) {done.get();}
	}
				
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {
		final Configuration conf = new Configuration();
		final CrossValidationEvaluation cv = new CrossValidationEvaluation(conf);
		final ExecutorService threadExecutor = cv.m_threadExecutor;
		// Passo all'esperimento lo stesso esecutore di quello in CV, in modo che accodino le esecuzioni assieme 
		final TestSetEvaluation pe = new TestSetEvaluation(conf, threadExecutor);
		// Cerco di recuperare i risultati CV gia' esistenti
		try {
			final List<String[]> winners = cv.getWinners();
			System.out.println("...Using pre-existent CV results");
			for (final Future<Void> done : pe.experiment(winners)) {done.get();}
		}
		// ...altrimenti eseguo l'esperimento in CV.
		catch (Exception e) {
			pe.incrementalExperiment(cv.experiment());
		}
		threadExecutor.shutdown();
	}
}
