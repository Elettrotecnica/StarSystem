package conf;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import utils.Utils;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

public class Configuration {
	
	public String m_fileSeparator;
	public int    m_nThreads;
	public Random m_random;
	
	public String[] m_datasetNames;
	public int      m_nDatasets;
	
	// Feature selections parameters
	public int[] m_featureSelectionNFeatures;
	public int   m_featureSelectionIterations;
	public int   m_featureSelectionFolds;
	public int   m_nMaxFeatures;
	
	public int   m_crossValidationIterations;
	public int   m_crossValidationFolds;
	
	public String   m_sortAttributeName;
	public String   m_sliceAttributeName;
	public String[] m_attributesToRemove;
	public String   m_evaluationMeasure;
	
	public String[] m_featureSelectionDatasetSlices;
	public String[] m_crossValidationDatasetSlices;
	public String[] m_testSetDatasetSlices;
	
	public String m_baseFolder;
	public String m_datasetFolder;
	public String m_featureSelectionFolder;
	public String m_crossValidationFolder;
	public String m_testSetFolder;
	public String m_resultsFolder;
	
//	public final String m_resultsFileName;
	
	public final List<Classifier>   m_classifiers = new ArrayList<Classifier>();
	public final List<ASEvaluation> m_filters     = new ArrayList<ASEvaluation>();
	public final List<Classifier>   m_wrappers    = new ArrayList<Classifier>();
	
	public int m_nFilters;
	public int m_nWrappers;
	public int m_nClassifiers;
	
	public String crossValidationSelectedDatasetPath() {
		final String[] pathTokens = {m_baseFolder, m_crossValidationFolder, m_datasetFolder, ""};
		return Utils.join(pathTokens, m_fileSeparator);
	}
	
	public String testSetSelectedDatasetPath() {
		final String[] pathTokens = {m_baseFolder, m_testSetFolder, m_datasetFolder, ""};
		return Utils.join(pathTokens, m_fileSeparator);
	}
	
	private void setDefaultParameters() {
		m_fileSeparator = File.separator;
		m_nThreads  = Runtime.getRuntime().availableProcessors();
		m_random    = new Random(System.currentTimeMillis());
		
//		String[] datasetNames = {"IT1"};
//		m_datasetNames = datasetNames;
//		m_nDatasets    = datasetNames.length;
//		
//		int[] featureSelectionNFeatures = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
//		Arrays.sort(featureSelectionNFeatures);
//		m_featureSelectionNFeatures  = featureSelectionNFeatures;
//		m_nMaxFeatures               = featureSelectionNFeatures[featureSelectionNFeatures.length-1];
		
		m_datasetNames = null;
		m_nDatasets    = 0;
		
		m_featureSelectionNFeatures  = null;
		m_nMaxFeatures               = 0;

		m_featureSelectionIterations = 10;
		m_featureSelectionFolds      = 10;
		
		m_crossValidationIterations = 10;
		m_crossValidationFolds      = 10;
		
//		m_sortAttributeName  = "date";
//		m_sliceAttributeName = "season";
		m_sortAttributeName  = null;
		m_sliceAttributeName = null;
		m_attributesToRemove = new String[0];

		// can choose between:
		// - accuracy
		// - precision
		// - recall
		// - f-measure
		m_evaluationMeasure  = "accuracy";
		
		String[] featureSelectionDatasetSlices = null;
		String[] crossValidationDatasetSlices  = null;
		String[] testSetDatasetSlices          = null;
//		String[] featureSelectionDatasetSlices = {"2011", "2012"};
//		String[] crossValidationDatasetSlices  = {"2011", "2012"};
//		String[] testSetDatasetSlices          = {"2013"};
		m_featureSelectionDatasetSlices = featureSelectionDatasetSlices;
		m_crossValidationDatasetSlices  = crossValidationDatasetSlices;
		m_testSetDatasetSlices          = testSetDatasetSlices;
		
		m_baseFolder             = ".";
		m_datasetFolder          = "01-dataset";
		m_featureSelectionFolder = "02-feature-selection";
		m_crossValidationFolder  = "03-cross-validation";
		m_testSetFolder          = "04-test-set";
		m_resultsFolder          = "05-results";
	}
	
	private void setDefaultClassifiers() {
		String options;
		
		m_classifiers.clear();
		
		// Classifiers
		final Classifier naiveBayes = new NaiveBayes();
		m_classifiers.add(naiveBayes);
		
		final Classifier j48 = new J48();
		m_classifiers.add(j48);
		
		final Classifier tan = new BayesNet();
		options = "-D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5";
		try {tan.setOptions(weka.core.Utils.splitOptions(options));} catch (Exception e) {e.printStackTrace();}
		m_classifiers.add(tan);
		
		final Classifier logistic = new Logistic();
		m_classifiers.add(logistic);
		
		final Classifier smo = new SMO();
		options = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.Puk -C 250007 -O 1.0 -S 1.0\"";
		try {smo.setOptions(weka.core.Utils.splitOptions(options));} catch (Exception e) {e.printStackTrace();}
		m_classifiers.add(smo);
		
		final Classifier simpleLogistic = new SimpleLogistic();
		m_classifiers.add(simpleLogistic);
		
		final Classifier randomForest10 = new RandomForest();
		options = "-I 10 -K 0 -S 1";
		try {randomForest10.setOptions(weka.core.Utils.splitOptions(options));} catch (Exception e) {e.printStackTrace();}
		m_classifiers.add(randomForest10);
		
		final Classifier randomForest50 = new RandomForest();
		options = "-I 50 -K 0 -S 1";
		try {randomForest50.setOptions(weka.core.Utils.splitOptions(options));} catch (Exception e) {e.printStackTrace();}
		m_classifiers.add(randomForest50);

//		Classifier libsvm = new LibSVM();
//		options = "-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -Z -seed 1";
//		try {libsvm.setOptions(weka.core.Utils.splitOptions(options));} catch (Exception e) {e.printStackTrace();}
//		CLASSIFIERS.add(libsvm);
		
		m_nClassifiers = m_classifiers.size();
	}
	
	
	private void setDefaultFilters() {			
		// Filters
		m_filters.clear();
		
		m_filters.add(new ChiSquaredAttributeEval());
		m_filters.add(new InfoGainAttributeEval());
		
		m_nFilters = m_filters.size();
	}
	
	private void setDefaultWrappers() {
		String options;
		
		m_wrappers.clear();
		
		// Wrappers
		final Classifier naiveBayesW = new NaiveBayes();
		m_wrappers.add(naiveBayesW);
		
		final Classifier j48W = new J48();
		m_wrappers.add(j48W);
		
		final Classifier simpleLogisticW = new SimpleLogistic();
		m_wrappers.add(simpleLogisticW);
		
		final Classifier smoW = new SMO();
		options = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.Puk -C 250007 -O 1.0 -S 1.0\"";
		try {smoW.setOptions(weka.core.Utils.splitOptions(options));} catch (Exception e) {e.printStackTrace();}
		m_wrappers.add(smoW);
		
//		final Classifier libsvm = new LibSVM();
//		options = "-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -Z -seed 1";
//		try {libsvm.setOptions(weka.core.Utils.splitOptions(options));} catch (Exception e) {e.printStackTrace();}
//		WRAPPERS.add(libsvm);
		
		m_nWrappers = m_wrappers.size();
	}
	
	public Configuration() {
		setDefaultParameters();
		setDefaultClassifiers();
		setDefaultFilters();
		setDefaultWrappers();
	}
	
	public Configuration(final String confFilePath) throws Exception {
		
		setDefaultParameters();
		
		final BufferedReader reader = new BufferedReader(new FileReader(confFilePath));
		String line = reader.readLine(); 
		while (line != null) {
			line = line.trim();
			if (line.equals("") || line.charAt(0) == '#' || line.charAt(0) == ';') {
				line = reader.readLine();
				continue;
			}
			
			final String param = line.substring(0, line.indexOf('=')).trim();
			final String value = line.substring(line.indexOf('=')+1).trim();

			String[] values;
			int nValues;
			
			String[] classTokens;
			String className;
			String[] classOptions;
			
		    switch (param) {
		    	case "separator":  
		    		m_fileSeparator = value;
		    		break;
		    	case "n_threads":  
	    			m_nThreads = Integer.parseInt(value);
		    		break;
		    	case "datasets":  
		    		values = value.split(",");
		    		nValues = values.length;
		    		m_datasetNames = new String[nValues];
		    		m_nDatasets = nValues;
		    		for (int i = 0; i < nValues; i++) {
	    				m_datasetNames[i] = values[i].trim(); 
	    			}
		    		break;
		    	case "n_feature_selection_features":
		    		values = value.split(",");
		    		nValues = values.length;
		    		m_featureSelectionNFeatures = new int[nValues];
		    		for (int i = 0; i < nValues; i++) {
		    			m_featureSelectionNFeatures[i] = Integer.parseInt(values[i].trim()); 
	    			}
		    		Arrays.sort(m_featureSelectionNFeatures);
		    		m_nMaxFeatures = m_featureSelectionNFeatures[nValues-1];
		    		break;
		    	case "n_feature_selection_iterations":  
		    		m_featureSelectionIterations = Integer.parseInt(value);
		    		break;
		    	case "n_feature_selection_folds":  
		    		m_featureSelectionFolds = Integer.parseInt(value);
		    		break;
		    	case "n_cross_validation_iterations":  
		    		m_crossValidationIterations = Integer.parseInt(value);
		    		break;
		    	case "n_cross_validation_folds":  
		    		m_crossValidationFolds = Integer.parseInt(value);
		    		break;
		    	case "attributes_to_remove":
		    		final String[] vals = value.split(","); 
		    		final int nVals = vals.length;
		    		final String[] attributesToRemove = new String[nVals];
		    		for (int i = 0; i < nVals; i++) {
		    			final String attribute = vals[i].trim();
		    			attributesToRemove[i] = attribute;
		    		}
		    		m_attributesToRemove = attributesToRemove;
		    		break;
		    	case "sort_attribute_name":  
		    		m_sortAttributeName = value;
		    		break;
		    	case "slice_attribute_name":  
		    		m_sliceAttributeName = value;
		    		break;
		    	case "evaluation_measure":  
		    		m_evaluationMeasure = value;
		    		break;
		    	case "feature_selection_dataset_slices":  
		    		values = value.split(",");
		    		nValues = values.length;
		    		m_featureSelectionDatasetSlices = new String[nValues];
		    		for (int i = 0; i < nValues; i++) {
		    			m_featureSelectionDatasetSlices[i] = values[i].trim(); 
	    			}
		    		break;
		    	case "cross_validation_dataset_slices":  
		    		values = value.split(",");
		    		nValues = values.length;
		    		m_crossValidationDatasetSlices = new String[nValues];
		    		for (int i = 0; i < nValues; i++) {
		    			m_crossValidationDatasetSlices[i] = values[i].trim(); 
	    			}
		    		break;
		    	case "test_set_dataset_slices":  
		    		values = value.split(",");
		    		nValues = values.length;
		    		m_testSetDatasetSlices = new String[nValues];
		    		for (int i = 0; i < nValues; i++) {
		    			m_testSetDatasetSlices[i] = values[i].trim(); 
	    			}
		    		break;
		    	case "base_folder":  
		    		m_baseFolder = value;
		    		break;
		    	case "dataset_folder":  
		    		m_datasetFolder = value;
		    		break;
		    	case "feature_selection_folder":  
		    		m_featureSelectionFolder = value;
		    		break;
		    	case "cross_validation_folder":  
		    		m_crossValidationFolder = value;
		    		break;
		    	case "test_set_folder":  
		    		m_testSetFolder = value;
		    		break;
		    	case "results_folder":  
		    		m_resultsFolder = value;
		    		break;
		    	case "classifier":  
		    		classTokens = value.split(" ");
		    		className = classTokens[0];
		    		if (classTokens.length > 1) {
			    		classOptions = weka.core.Utils.splitOptions(
			    				value.substring(value.indexOf(" ")+1, value.length()).trim());
		    		} else {
		    			classOptions = null;
		    		}
		    		m_classifiers.add(Classifier.forName(className, classOptions));
		    		break;
		    	case "wrapper":
		    		classTokens = value.split(" ");
		    		className = classTokens[0];
		    		if (classTokens.length > 1) {
			    		classOptions = weka.core.Utils.splitOptions(
			    				value.substring(value.indexOf(" ")+1, value.length()).trim());
		    		} else {
		    			classOptions = null;
		    		}
		    		m_wrappers.add(Classifier.forName(className, classOptions));
		    		break;
		    	case "filter":  
		    		classTokens = value.split(" ");
		    		className = classTokens[0];
		    		if (classTokens.length > 1) {
			    		classOptions = weka.core.Utils.splitOptions(
			    				value.substring(value.indexOf(" ")+1, value.length()).trim());
		    		} else {
		    			classOptions = null;
		    		}
		    		m_filters.add(ASEvaluation.forName(className, classOptions));
		    		break;
		    }
		    line = reader.readLine(); 
		}
		reader.close();
		
		m_nFilters = m_filters.size();
		if (m_nFilters == 0) {
			System.out.print("WARNING: no filters specified. They won't be applied to feature selection.");
		}
		m_nWrappers = m_wrappers.size();
		if (m_nWrappers == 0) {
			System.out.print("WARNING: no wrappers specified. They won't be applied to feature selection.");
		}
		m_nClassifiers = m_classifiers.size();
		if (m_nClassifiers == 0) {
			setDefaultClassifiers();
			System.out.print("WARNING: no classifiers specified. Using defaults.");
		}
		
		if (m_datasetNames == null) {
			throw new Exception("No 'dataset' specified in configuration");
		}		
		if (m_featureSelectionNFeatures == null) {
			throw new Exception("No 'n_feature_selection_features' specified in configuration");
		}
		if (m_sortAttributeName == null) {
			throw new Exception("No 'sort_attribute_name' specified in configuration");
		}
		if (m_sliceAttributeName == null) {
			throw new Exception("No 'slice_attribute_name' specified in configuration");
		}
		if (m_featureSelectionDatasetSlices == null) {
			throw new Exception("No 'feature_selection_dataset_slices' specified in configuration");
		}
		if (m_crossValidationDatasetSlices == null) {
			throw new Exception("No 'cross_validation_dataset_slices' specified in configuration");
		}
		if (m_testSetDatasetSlices == null) {
			throw new Exception("No 'test_set_dataset_slices' specified in configuration");
		}
	}
}