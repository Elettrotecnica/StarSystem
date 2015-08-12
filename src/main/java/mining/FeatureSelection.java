package mining;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import conf.Configuration;

import utils.Utils;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class FeatureSelection {
	
	final Configuration m_conf;
	
	final String m_resultsPath;
	
	final ExecutorService m_threadExecutor;
	
	final String m_filtersNames;
	final String m_wrappersNames;
	
	public FeatureSelection() {
		this(new Configuration());
	}
	
	public FeatureSelection(Configuration conf) {
		m_conf = conf;

		m_threadExecutor = Executors.newFixedThreadPool(m_conf.m_nThreads);
		
		// Build filter names from conf
		final int nFilters = m_conf.m_filters.size();
		final int lastFilter = nFilters-1;
		String filtersNames = "";
		for (int i = 0; i < nFilters; i++) {
			ASEvaluation filter = m_conf.m_filters.get(i);
			if (i == 0) {filtersNames+="_filters";}
			filtersNames+=getFilterName(filter);
			if (i != lastFilter) {
				filtersNames+="-";
			}
		}
		m_filtersNames = filtersNames;

		// Build wrapper names from conf
		final int nWrappers = m_conf.m_wrappers.size();
		final int lastWrapper = nWrappers-1;
		String wrappersNames = "";
		for (int i = 0; i < nWrappers; i++) {
			final Classifier w = m_conf.m_wrappers.get(i);
			if (i == 0) {wrappersNames+="_wrappers";}
			wrappersNames+=getWrapperName(w);
			if (i != lastWrapper) {
				wrappersNames+="-";
			}
		}
		m_wrappersNames = wrappersNames;
		
		m_resultsPath = resultsPath();
		Utils.requireDir(m_resultsPath);
	}
	
	
	private String datasetFile(final String competition) {
		final String[] pathTokens = {m_conf.m_baseFolder, m_conf.m_datasetFolder, m_conf.m_featureSelectionFolder, competition + ".arff"};
		return Utils.join(pathTokens, m_conf.m_fileSeparator);
	}
	
	private String resultsPath() {
		final String[] pathTokens = {m_conf.m_baseFolder, m_conf.m_featureSelectionFolder, m_conf.m_resultsFolder, ""};
		return Utils.join(pathTokens, m_conf.m_fileSeparator);
	}
	
	private Instances getCompetitionDataset(final String competition) {
		return Utils.readFile(datasetFile(competition));
	}
	
	public class Exp {
		final protected AttributeSelection m_meta;
		final protected Instances m_data;
		
		public Exp(final Instances data, final int nFolds) {
			m_data = data;
			m_meta = new AttributeSelection();
			m_meta.setSeed(m_conf.m_random.nextInt());
			m_meta.setFolds(nFolds);
		}
		
		protected double[][] getRankedAttributes() throws Exception {
			m_meta.SelectAttributes(m_data);
			final double[][] rankedAttributes = m_meta.rankedAttributes(); 
			Utils.sortByValue(rankedAttributes);
			return rankedAttributes;
		}
	}
	
	public class FilterExp extends Exp {
		public FilterExp(
				final Instances data, 
				final ASEvaluation eval, 
				final int nFolds) {
			super(data, nFolds);
			m_meta.setEvaluator(eval);
			m_meta.setSearch(new Ranker());
		}
	}
	
	public class WrapperExp extends Exp {
		public WrapperExp(
				final Instances data, 
				final Classifier classifier, 
				final int nFolds) throws Exception {
			super(data, nFolds);
			
			String options;
			
			final WrapperSubsetEval eval = new WrapperSubsetEval();
			options = "-F 5 -T 0.01";
			eval.setSeed(m_conf.m_random.nextInt());
			eval.setOptions(weka.core.Utils.splitOptions(options));
			eval.setClassifier(classifier);
			m_meta.setEvaluator(eval);
			
			final GreedyStepwise search = new GreedyStepwise();
			options = "-R -T -1.7976931348623157E308 -N -1";
			search.setOptions(weka.core.Utils.splitOptions(options));
			m_meta.setSearch(search);
		}
	}
		
	private String getFilterName(final ASEvaluation filter) {
		final String[] filterNameTokens = filter.getClass().getName().split("\\.");
		return filterNameTokens[filterNameTokens.length - 1].replaceAll("\\.", "_");
	}
	
	private String getWrapperName(final Classifier wrapper) {
		final String[] wrapperNameTokens = wrapper.getClass().getName().split("\\.");
		final String wrapperName = wrapperNameTokens[wrapperNameTokens.length - 1];
// If filename is too big, execution will fail. For now we stick to a short filename.
//		for (String option : wrapper.getOptions()) {
//			option = option.replaceAll("\\.", "-");
//			wrapperName+=option;
//		}; 
		return wrapperName;
	}
	
	private String getOutFileName(final String dataset, final int nFeatures) {
		return m_resultsPath + m_conf.m_fileSeparator 
		+ dataset 
		+ m_filtersNames
		+ m_wrappersNames
		+ "_fold"       + m_conf.m_featureSelectionFolds
		+ "_iterations" + m_conf.m_featureSelectionIterations
		+ "_features"   + nFeatures
		+ ".arff";
	}
	
	public void selection() throws Exception {
		final List<Instances> filteredInstances = new ArrayList<Instances>(m_conf.m_nDatasets);
		final List<List<Future<double[][]>>> competitionWrappersRankings = new ArrayList<List<Future<double[][]>>>(m_conf.m_nDatasets);
		
		for(final String dataset: m_conf.m_datasetNames) {
			Instances data = getCompetitionDataset(dataset);
						
			final int nFiltersRankings = m_conf.m_nFilters * m_conf.m_featureSelectionIterations;
			final List<Future<double[][]>> filtersRankingsList = new ArrayList<Future<double[][]>>(nFiltersRankings);
			for (final ASEvaluation filter : m_conf.m_filters) {
				// We use a copy of the filter for thread safety
				for (final ASEvaluation filterCopy : ASEvaluation.makeCopies(filter, m_conf.m_featureSelectionIterations)) {
					// We use a copy of the dataset for thread safety
					final Instances dataCopy = new Instances(data);
					filtersRankingsList.add(m_threadExecutor.submit(new Callable<double[][]>() {
					    public double[][] call() throws Exception {
							final FilterExp fexp = new FilterExp(dataCopy, filterCopy, m_conf.m_featureSelectionFolds);
					    	System.out.println("Filter start");
					    	final double[][] ret = fexp.getRankedAttributes();
					    	System.out.println("Filter end");
					    	return ret;
					    }
					}));
				}
			}
			
			if (filtersRankingsList.size() > 0) {
				data = Utils.reorderInstances(data, filtersRankingsList);
				filtersRankingsList.clear();
				data = Utils.keepNFeatures(data, m_conf.m_nMaxFeatures);
				filteredInstances.add(data);
			}
			
			final int nWrappersRankings = m_conf.m_nWrappers * m_conf.m_featureSelectionIterations;
			final List<Future<double[][]>> wrappersRankingsList = new ArrayList<Future<double[][]>>(nWrappersRankings);
			for (final Classifier wrapper : m_conf.m_wrappers) {
				// We use a copy of the classifier for thread safety
				for (final Classifier wrapperCopy : Classifier.makeCopies(wrapper, m_conf.m_featureSelectionIterations)) {
					// We use a copy of the dataset for thread safety
					final Instances dataCopy = new Instances(data);
					wrappersRankingsList.add(m_threadExecutor.submit(new Callable<double[][]>() {
						public double[][] call() throws Exception {
							final WrapperExp wexp = new WrapperExp(dataCopy, wrapperCopy, m_conf.m_featureSelectionFolds);
							System.out.println("Wrapper start");
					    	final double[][] ret = wexp.getRankedAttributes();
							System.out.println("Wrapper end");
					    	return ret;
					    }
					}));
				}
			}
			competitionWrappersRankings.add(wrappersRankingsList);
		}

		for(int i = 0; i < m_conf.m_nDatasets; i++) {
			final String dataset = m_conf.m_datasetNames[i];
			final List<Future<double[][]>> wrappersRankingsList = competitionWrappersRankings.get(i);

			Instances data = filteredInstances.get(i);
//			Utils.saveInstances(data, File.createTempFile("filtersDataset", ".arff"));
			if (wrappersRankingsList.size() > 0) {
				data = Utils.reorderInstances(data, wrappersRankingsList);
				wrappersRankingsList.clear();
			}
			
			for (final int nFeature: m_conf.m_featureSelectionNFeatures) {
				final String outFileName = getOutFileName(dataset, nFeature);
				System.out.println(outFileName);
				Utils.saveInstances(Utils.keepNFeatures(data, nFeature), outFileName);
			}
		}

		filteredInstances.clear();
		competitionWrappersRankings.clear();
				
		m_threadExecutor.shutdown();
	}
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {
		new FeatureSelection().selection();
	}

}
