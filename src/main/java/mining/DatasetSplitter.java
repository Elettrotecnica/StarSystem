package mining;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.text.ParseException;
import java.util.HashMap;

import conf.Configuration;

import utils.Utils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

// This object cares to split dataset by the split attribute
// defined in configuration.
// Once splitted, dataset is assigned to the 3 phases of the analysis:
// 1) Feature selection
// 2) Cross Validation evaluation
// 3) Test Set evaluation
public class DatasetSplitter {
	
	// Object holding all configuration so they can be
	// shared during all the analysis
	final Configuration m_conf;
	
	public DatasetSplitter() {
		this(new Configuration());
	}
	
	public DatasetSplitter(final Configuration conf) {
		m_conf = conf;
	}
	
	//
	//// Functions used to retrieve file paths
	//
	
	private String datasetFile(final String datasetName) {
		final String[] pathTokens = {m_conf.m_baseFolder, m_conf.m_datasetFolder, m_conf.m_datasetFolder, datasetName + ".arff"};
		return Utils.join(pathTokens, m_conf.m_fileSeparator);
	}
	
	private String featureSelectionDatasetFile(final String datasetName) {
		final String[] pathTokens = {m_conf.m_baseFolder, m_conf.m_datasetFolder, m_conf.m_featureSelectionFolder, datasetName + ".arff"};
		return Utils.join(pathTokens, m_conf.m_fileSeparator);
	}

	private String crossValidationDatasetFile(final String datasetName) {
		final String[] pathTokens = {m_conf.m_baseFolder, m_conf.m_datasetFolder, m_conf.m_crossValidationFolder, datasetName + ".arff"};
		return Utils.join(pathTokens, m_conf.m_fileSeparator);
	}

	private String testSetDatasetFile(final String datasetName) {
		final String[] pathTokens = {m_conf.m_baseFolder, m_conf.m_datasetFolder, m_conf.m_testSetFolder, datasetName + ".arff"};
		return Utils.join(pathTokens, m_conf.m_fileSeparator);
	}

	private String featureSelectionResultsPath() {
		final String[] pathTokens = {m_conf.m_baseFolder, m_conf.m_featureSelectionFolder, m_conf.m_resultsFolder, ""};
		return Utils.join(pathTokens, m_conf.m_fileSeparator);
	}
		
	////
	
	
	// Slices dataset by separating instances by the distinct values of attribute
	// which name is held into conf's 'm_sliceAttributeName' parameter.
	private HashMap<String,Instances> sliceByAttribute(final Instances instances) {
		final HashMap<String,Instances> slices = new HashMap<String,Instances>();
		final Attribute sliceAttribute = instances.attribute(m_conf.m_sliceAttributeName);	
		final int nInstances = instances.numInstances();
		for (int j = 0; j < nInstances; j++) {
			final Instance in = instances.instance(j);
			final String sliceName = in.toString(sliceAttribute);
			final Instances slice;
			if (!slices.containsKey(sliceName)) {
				slice = new Instances(instances, 0);
				slices.put(sliceName, slice);
			} else {
				slice = slices.get(sliceName);
			}
			slice.add(in);
		}
		return slices;
	}
	
	private HashMap<String,Instances> slice(final Instances instances) {
		return sliceByAttribute(instances);
	}
	
	// Dataset is sliced, then each slice is assigned to one of the phases of the analysis
	// by looking into conf what slice was to be given to what of the phases.
	private Instances[] assignSlices(final Instances instances) throws ParseException {
		final HashMap<String,Instances> slices = slice(instances);
		final Instances[] splittedDataset = new Instances[3];
		
		splittedDataset[0] = new Instances(instances, 0);
		for (final String key : m_conf.m_featureSelectionDatasetSlices) {
			if (slices.containsKey(key)) {
				Utils.addInstancesToDataset(slices.get(key), splittedDataset[0]);
			}
		}
		splittedDataset[1] = new Instances(instances, 0);
		for (final String key : m_conf.m_crossValidationDatasetSlices) {
			if (slices.containsKey(key)) {
				Utils.addInstancesToDataset(slices.get(key), splittedDataset[1]);
			}
		}
		splittedDataset[2] = new Instances(instances, 0);
		for (final String key : m_conf.m_testSetDatasetSlices) {
			if (slices.containsKey(key)) {
				Utils.addInstancesToDataset(slices.get(key), splittedDataset[2]);
			}
		}
		return splittedDataset;
	}
	
	// Sort dataset by the column specified for sotring into confs
	private Instances sort(final Instances dataset) throws Exception {
		dataset.sort(dataset.attribute(m_conf.m_sortAttributeName).index());
		return dataset;
	}

	// Remove sorting and slicing attributes which are useless for the analysis and could
	// lead to data-snooping (as they usually are somewhat temporal values)
	private Instances removeSnoopingAttributes(final Instances dataset) throws Exception {
		return Utils.removeAttribute(Utils.removeAttribute(dataset, 
				m_conf.m_sliceAttributeName), m_conf.m_sortAttributeName);
	}

	// Slice and assign the dataset to the phases, then save the 3 datasets in their proper files.
	public void buildDatasets() throws Exception {
		for (final String datasetName: m_conf.m_datasetNames) {
			final String datasetFile = datasetFile(datasetName);
			Instances dataset = Utils.readFile(datasetFile);
			// Remove unused attributes
			for (final String attToRemove : m_conf.m_attributesToRemove) {
				if (dataset.attribute(attToRemove) != null)
					dataset = Utils.removeAttribute(dataset,attToRemove);
			}
			final Instances[] splittedDataset = assignSlices(dataset);
			Utils.saveInstances(removeSnoopingAttributes(sort(splittedDataset[0])), featureSelectionDatasetFile(datasetName));
			Utils.saveInstances(removeSnoopingAttributes(sort(splittedDataset[1])), crossValidationDatasetFile(datasetName));
			Utils.saveInstances(removeSnoopingAttributes(sort(splittedDataset[2])), testSetDatasetFile(datasetName));
		}
	}
	
	// Obtain a HashMap containing all attributes into a reference dataset
	private HashMap<String,Boolean> getAttributeReference(final Instances reference) {
		final HashMap<String,Boolean> map = new HashMap<String,Boolean>();
		final int nAttributes = reference.numAttributes();
		for (int i = 0; i < nAttributes; i++) {
			final String aName = reference.attribute(i).name();
			map.put(aName, true);
		} ; return map;
	}

	// Uses a reference built by 'getAttributeReference' on a dataset to it 
	// will contains all and only the atrributes specified in the reference
	private Instances attributesFromReference(
			final Instances dataset, 
			final HashMap<String,Boolean> reference) throws Exception {
		final int nAttributes = dataset.numAttributes();
		String keptAttributes = "";
		for (int i = 0; i < nAttributes; i++) {
			final String aName = dataset.attribute(i).name();
			if (reference.containsKey(aName)) {
				if (!keptAttributes.equals("")) {
					keptAttributes+=",";}
				keptAttributes+="" + (i + 1);
			}
		}
		
		// I will keep only the attributes which are NOT in the ones I saved
		final Remove remove = new Remove();
		remove.setAttributeIndices(keptAttributes);
		remove.setInvertSelection(true);
		final Instances result = Utils.useFilter(dataset, remove);
		
		// For safety, we leave from here only if reference 
		// and dataset have the same number of attributes now
		assert(result.numAttributes() == reference.size() + 1);
		
		return result;
	}
	
	// Uses the results of feature selection on datasets assigned to cross validation and
	// test set evaluation, so their attributes will be filtered according to selection.
	void datasetAttributesFromReference() throws Exception {
		for(final String datasetName: m_conf.m_datasetNames) {
			final String regexp = "^" + datasetName + ".*.arff$";
			final FilenameFilter fileFilter = Utils.getFileFilter(regexp);
			
			// Dataset with all the features for the following phases
			final Instances CVDataset   = Utils.readFile(crossValidationDatasetFile(datasetName));
			final Instances testDataset = Utils.readFile(testSetDatasetFile(datasetName));
			
			// Take the folder with feature selection results...
	        final File dir = new File(featureSelectionResultsPath());
	        final File[] datasetFiles = dir.listFiles(fileFilter);
			for (final File featureSet: datasetFiles) {
				if (featureSet.isFile() && !featureSet.isHidden()) {
					final String featureSetDescription = featureSet.getName();
					// ...use them as reference for attributes...
					final HashMap<String,Boolean> reference = getAttributeReference(Utils.readFile(featureSet));
					Instances modifiedDataset;
					
					// ...so we can apply selection on cross validation dataset...
					modifiedDataset = attributesFromReference(CVDataset, reference);
					try {Utils.saveInstances(modifiedDataset, m_conf.crossValidationSelectedDatasetPath() + featureSetDescription);}
					catch (IOException e) {e.printStackTrace();}
					
					// ...and test set one.
					modifiedDataset = attributesFromReference(testDataset, reference);
					try {Utils.saveInstances(modifiedDataset, m_conf.testSetSelectedDatasetPath() + featureSetDescription);}
					catch (IOException e) {e.printStackTrace();}
				}
			}
		}
	}
			
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {
		final DatasetSplitter exp = new DatasetSplitter();
		exp.buildDatasets();
		exp.datasetAttributesFromReference();
	}

}
