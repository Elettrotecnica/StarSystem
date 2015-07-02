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
import weka.classifiers.rules.ZeroR;
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

// Classe che valuta i modelli ricevuti in input sui dataset 
// suddivisi per numero di feature, restituendo tutte le combinazioni 
// campionato/numero di feature/modello che superano la baseline ZeroR
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
	}    

	private File getOutputFile(final String dataset, final String nFeatures) {
		final String[] pathTokens = {m_resultsFolder, dataset + "-" + nFeatures + "featuresCVResults.arff"};
		return new File(Utils.join(pathTokens, m_conf.m_fileSeparator));
	}
	
	// Il PairedTTester di weka mi toglie il namespace dalle classi nei risultati.
	// Lo devo recuperare dal file originale in questo modo.
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
	    // Leggo i risultati dell'esperimento...
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

	    // Recupero il risultato della baseline
    	final String baselineStdDev = Double.toString(matrix.getStdDev(0, 0));
    	final String baselineMean   = Double.toString(matrix.getMean(0, 0));

	    // Scorro i risultati per ciascun modello...
	    final List<String[]> winners = new ArrayList<String[]>();
	    for (int i = 0; i < nMethods; i++) {
	    	// ...se ho significativamente superato la baseline...
	    	if (matrix.getSignificance(i, 0) == ResultMatrix.SIGNIFICANCE_WIN) {
		    	final String stdDev = Double.toString(matrix.getStdDev(i, 0));
		    	final String mean   = Double.toString(matrix.getMean(i, 0));
		    	final String[] method = methodsOriginalNames[i];
	    		final String className = method[0];
	    		final String options   = method[1];
		    	// ...restituisco i modelli vincitori in uscita.
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
		
        // Imposto i dataset. Faccio subito perche' se non ne trovo ho gia' finito.
//        final DefaultListModel<File> model = new DefaultListModel<File>();
        final DefaultListModel<File> model = new DefaultListModel<File>();
        final String regexp = "^" + competition + ".*features" + nFeatures + ".arff$";
		final FilenameFilter fileFilter = Utils.getFileFilter(regexp);
        final File dir = new File(m_datasetFolder);
        // Aggiungo i dataset all'esperimento.
        for (final File train : dir.listFiles(fileFilter)) {model.addElement(train);}
        if (model.size() == 0) return null;
        exp.setDatasets(model);
		
		// Imposto i classificatori dell'esperimento
		// aggiungo la baseline ZeroR come primo classificatore 
		// in modo che sia considerato la baseline di default
		models.add(0, new ZeroR());
		exp.setPropertyArray(models.toArray());
		exp.setUsePropertyIterator(true);
		
		// Imposto l'esperimento come classificazione
		final ClassifierSplitEvaluator se = new ClassifierSplitEvaluator();
		final Class<? extends ClassifierSplitEvaluator> seClass = se.getClass();
	    
	    // Imposto l'esperimento in cross validation
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
        
        // Imposto il numero di iterazioni
        exp.setRunLower(1);
        exp.setRunUpper(m_conf.m_crossValidationIterations);
        		
		// Imposto il file di output dell'esperimento
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
		// Accodo l'esperimento nell'esecutore
		return m_threadExecutor.submit(new Callable<List<String[]>>() {
			public List<String[]> call() throws Exception {
				// Uso copie dei classificatori per la thread safety
				final List<Classifier> modelsCopy = new ArrayList<Classifier>(m_conf.m_nClassifiers);
				for (final Classifier model: m_conf.m_classifiers) {modelsCopy.add(Classifier.makeCopy(model));}
				return exp(dataset, nFeatures, modelsCopy);
			}
		});
	}
	
	// Il risultato dell'esperimento in CV e' una lista di future: non faccio aspettare l'esecuzione qui,
	// ma passo direttamente la parola al consumatore dei risultati (leggasi - la valutazione sul test set -).
	// In questo modo evito che il sistema resti senza nulla da fare mentre aspetta gli ultimi esperimenti in CV
	// e vada direttamente ad iniziare le valutazioni sul test set che puo' gia' fare.
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
