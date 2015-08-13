package mining;

import java.io.File;

import conf.Configuration;

// All phases of the analysis
public class Analysis {
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {
		if (args.length < 1) {
			System.out.println("You must specify a configuration file");
			return;
		}
		final String confFilePath = args[0];
		if (!new File(confFilePath).canRead()) {
			throw new Exception("Specified configuration file doesn not exist or is not readable");
		}
		
		final Configuration conf = new Configuration(confFilePath);
		final DatasetSplitter sp = new DatasetSplitter(conf);
		
		sp.buildDatasets();
		if (conf.m_doFeatureSelection) {
			new FeatureSelection(conf).selection();
		} else {
			System.out.println("WARNING: reusing previous feature selection results");
		}
		sp.datasetAttributesFromReference();
		new ModelSelection(conf).selection();
	}
}
