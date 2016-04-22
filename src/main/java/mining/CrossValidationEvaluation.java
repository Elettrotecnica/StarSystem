package mining;

import java.beans.IntrospectionException;
import java.beans.PropertyDescriptor;
import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.swing.DefaultListModel;

import conf.Configuration;

import utils.Utils;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Range;
import weka.experiment.ClassifierSplitEvaluator;
import weka.experiment.CrossValidationResultProducer;
import weka.experiment.Experiment;
import weka.experiment.InstancesResultListener;
import weka.experiment.PairedCorrectedTTester;
import weka.experiment.PropertyNode;
import weka.experiment.ResultMatrix;
import weka.experiment.ResultMatrixPlainText;

// Class evaluating every combination of features returned by feature 
// selection against all the models configured in the experiment.
// Its result will be a set of candidate configurations to be tested on the test set.
public class CrossValidationEvaluation {
	
	final Configuration m_conf;
	
	final public ExecutorService m_threadExecutor;
	
	final private String m_datasetFolder;
	final private String m_resultsFolder;
	

	public CrossValidationEvaluation() {
		this(new Configuration());
	}
	
	public CrossValidationEvaluation(Configuration conf) {
		m_conf = conf;
		
		m_threadExecutor = Executors.newFixedThreadPool(m_conf.m_nThreads);
		
		final String[] datasetFolderTokens = {m_conf.m_baseFolder, m_conf.m_crossValidationFolder, m_conf.m_datasetFolder, ""};
		m_datasetFolder = Utils.join(datasetFolderTokens, m_conf.m_fileSeparator);
		
		final String[] resultsFolderTokens = {m_conf.m_baseFolder, m_conf.m_crossValidationFolder, m_conf.m_resultsFolder, ""};
		m_resultsFolder = Utils.join(resultsFolderTokens, m_conf.m_fileSeparator);
		
		Utils.requireDir(m_datasetFolder, false);
		// Empty results folder only if we are performing cross validation
		Utils.requireDir(m_resultsFolder, m_conf.m_doCrossValidation);
	}    

	private File getOutputFile(final String dataset, final String nFeatures) {
		final String[] pathTokens = {m_resultsFolder, dataset + "-" + nFeatures + "featuresCVResults.arff"};
		return new File(Utils.join(pathTokens, m_conf.m_fileSeparator));
	}
	
	// Weka's PairedTTester removes class's namespace from its results.
	// As a workaround I have to recover it using this method.
	private String[][] getClassifiers (final Instances results) {
		final Attribute classifiersAttr = results.attribute("Key_Scheme");
		final Attribute optionsAttr     = results.attribute("Key_Scheme_options");
		 
		final Enumeration<Instance> ri = results.enumerateInstances();
		final List<String[]> classifiers = new ArrayList<String[]>();
		String lastKey = null; while (ri.hasMoreElements()) {
			final Instance r = ri.nextElement();
			final String className = r.stringValue(classifiersAttr);
			final String options   = r.stringValue(optionsAttr);
			final String thisKey = className + options;
			final String[] key = {className, options};
			if (lastKey == null || !lastKey.equals(thisKey)) {
				classifiers.add(key);
				lastKey = thisKey;
			}
		}
		return classifiers.toArray(new String[0][0]);
	}
	
	private List<String[]> getWinners(final String competition, final String nFeatures, final File outFile) throws Exception {
		// Read experiment's results
	    final PairedCorrectedTTester tester = new PairedCorrectedTTester();
	    final Instances result = Utils.readFile(outFile);
	    tester.setInstances(result);
	    tester.setSortColumn(-1);
	    tester.setRunColumn(result.attribute("Key_Run").index());
	    tester.setFoldColumn(result.attribute("Key_Fold").index());
	    tester.setResultsetKeyColumns(
	    	new Range(
			    + (result.attribute("Key_Scheme").index() + 1)
			    + ","
			    + (result.attribute("Key_Scheme_options").index() + 1)
			    + ","
			    + (result.attribute("Key_Scheme_version_ID").index() + 1)));
	    tester.setDatasetKeyColumns(
    		new Range("" + (result.attribute("Key_Dataset").index() + 1)));
	    
	    final ResultMatrix matrix = new ResultMatrixPlainText();
	    matrix.setRemoveFilterName(false);
	    tester.setResultMatrix(matrix);
	    
	    tester.setDisplayedResultsets(null);
	    tester.setSignificanceLevel(0.05);
	    tester.setShowStdDevs(true);
	    
	    final String evalAttribute;
	    switch (m_conf.m_evaluationMeasure) {
	    	case "accuracy":  
	    		evalAttribute = "Percent_correct";
	    		break;
	    	case "precision":  
    			evalAttribute = "IR_precision";
	    		break;
	    	case "recall":  
    			evalAttribute = "IR_recall";
	    		break;
	    	case "f-measure":  
    			evalAttribute = "F_measure";
	    		break;
	    	default :  
    			evalAttribute = "Percent_correct";
    			System.out.println("WARNING Unrecognized evaluation mesure. Using Accuracy as default.");
	    		break;
	    }
	    
	    final int comparationColumnIndex = result.attribute(evalAttribute).index();
	    tester.multiResultsetFull(0, comparationColumnIndex);
	    
	    final String[][] methodsOriginalNames = getClassifiers(result);
	    final int nMethods = methodsOriginalNames.length;

	    // Get baseline reults
    	final String baselineStdDev = Double.toString(matrix.getStdDev(0, 0));
    	final String baselineMean   = Double.toString(matrix.getMean(0, 0));

	    // Foreach model's results
	    final List<String[]> winners = new ArrayList<String[]>();
	    for (int i = 0; i < nMethods; i++) {
	    	// ...if baseline was significantly outperformed...
	    	if (matrix.getSignificance(i, 0) == ResultMatrix.SIGNIFICANCE_WIN) {
		    	final String stdDev = Double.toString(matrix.getStdDev(i, 0));
		    	final String mean   = Double.toString(matrix.getMean(i, 0));
		    	final String[] method = methodsOriginalNames[i];
	    		final String className = method[0];
	    		final String options   = method[1];
		    	// ...forward this configuration to the next stage. 
		    	final String[] r = {competition, nFeatures, className, options, mean, stdDev, baselineMean, baselineStdDev};
		    	winners.add(r);
	    	}
	    }
	    
	    return winners;
	}
	
	private List<String[]> getWinners(final String competition, final String nFeatures) throws Exception {
		return getWinners(competition, nFeatures, getOutputFile(competition, nFeatures));
	}
	
	public List<String[]> getWinners() throws Exception {
		final int minNExp = m_conf.m_nDatasets * m_conf.m_featureSelectionNFeatures.length;
		final List <String[]> winners = new ArrayList<String[]>(minNExp);
		for (final String dataset : m_conf.m_datasetNames) {
			for (final int nFeature : m_conf.m_featureSelectionNFeatures) {
				for (final String[] winner : getWinners(dataset, Integer.toString(nFeature))) {
					winners.add(winner);
				}}} ; return winners;
	}
	
	private List<String[]> exp(final String competition, final String nFeatures, final List<Classifier> models) throws Exception {
		final Experiment exp = new Experiment();
		
		// Read datasets from files...
        final DefaultListModel<File> model = new DefaultListModel<File>();
        final String regexp = "^" + competition + ".*features" + nFeatures + ".arff$";
		final FilenameFilter fileFilter = Utils.getFileFilter(regexp);
        final File dir = new File(m_datasetFolder);
        // ...and include them in the experiment.
        for (final File train : dir.listFiles(fileFilter)) {model.addElement(train);}
        if (model.size() == 0) return null;
        exp.setDatasets(model);
		
        // Set classifiers for the experiment. Classifier n. 0 is the baseline
		models.add(0, Classifier.makeCopy(m_conf.m_baselineClassifier));
		exp.setPropertyArray(models.toArray());
		exp.setUsePropertyIterator(true);
		
		// Set the experiment as a classification experiment
		final ClassifierSplitEvaluator se = new ClassifierSplitEvaluator();
		final Class<? extends ClassifierSplitEvaluator> seClass = se.getClass();
	    
	    // Set the experiment as cross-validated and set number of folds
        final CrossValidationResultProducer cvrp = new CrossValidationResultProducer();
        final Class<? extends CrossValidationResultProducer> cvrpClass = cvrp.getClass();
        cvrp.setNumFolds(m_conf.m_crossValidationFolds);
        cvrp.setSplitEvaluator(se);
	        
        final PropertyNode[] propertyPath = new PropertyNode[2];
        try {
        	propertyPath[0] = new PropertyNode(se, new PropertyDescriptor("splitEvaluator", cvrpClass), cvrpClass);
        	propertyPath[1] = new PropertyNode(se.getClassifier(), new PropertyDescriptor("classifier", seClass), seClass);
        } catch (IntrospectionException e) {e.printStackTrace();}
	        
        exp.setResultProducer(cvrp);
        exp.setPropertyPath(propertyPath);
        
        // Set number of iterations for each experiment
        exp.setRunLower(1);
        exp.setRunUpper(m_conf.m_crossValidationIterations);
        		
		// Set output file for the experiment
		final InstancesResultListener irl = new InstancesResultListener();
		final File outFile = getOutputFile(competition, nFeatures);
		if (outFile.exists()) {outFile.delete();}
		outFile.createNewFile();
	    irl.setOutputFile(outFile);
	    exp.setResultListener(irl);
	    
	    exp.initialize();
	    exp.runExperiment();
	    exp.postProcess();
        
	    return getWinners(competition, nFeatures, outFile);
	}
	
	private Future<List<String[]>> submitExp(final String dataset, final String nFeatures) throws Exception {
		// Queue experiments in the executor
		return m_threadExecutor.submit(new Callable<List<String[]>>() {
			public List<String[]> call() throws Exception {
				// To ensure thread safety, always copy the classifier, instead of passing by reference!
				final List<Classifier> modelsCopy = new ArrayList<Classifier>(m_conf.m_nClassifiers);
				for (final Classifier model: m_conf.m_classifiers) {modelsCopy.add(Classifier.makeCopy(model));}
				return exp(dataset, nFeatures, modelsCopy);
			}
		});
	}
	
	// CV experiments returns futures to the next phase: this way we don't have to wait for 
	// every CV experiment to complete, for testing results we already have on the test set.
	public List<Future<List<String[]>>> experiment() throws Exception {
		final int nExperiments = m_conf.m_featureSelectionNFeatures.length * m_conf.m_nDatasets;
		final List<Future<List<String[]>>> results = new ArrayList<Future<List<String[]>>>(nExperiments);
		
		for (final int nFeatures : m_conf.m_featureSelectionNFeatures) {
			for (final String dataset : m_conf.m_datasetNames) {
				results.add(submitExp(dataset, Integer.toString(nFeatures)));
			}
		}
		return results;
	}
				
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {
		CrossValidationEvaluation cv = new CrossValidationEvaluation();
		cv.experiment();
		cv.m_threadExecutor.shutdown();
	}

}
