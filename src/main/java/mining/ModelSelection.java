package mining;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import conf.Configuration;


public class ModelSelection {
	
	final Configuration m_conf;
	
	public ModelSelection(Configuration conf) {
		m_conf = conf;
	}
	
	public ModelSelection() {
		this(new Configuration());
	}
	
	void selection() throws Exception {
		final CrossValidationEvaluation cv = new CrossValidationEvaluation(m_conf);
		final ExecutorService threadExecutor = cv.m_threadExecutor;
		final TestSetEvaluation te = new TestSetEvaluation(m_conf, threadExecutor);
		if (m_conf.m_doCrossValidation) {
			te.incrementalExperiment(cv.experiment());
		} else {
			System.out.println("WARNING: reusing previous cross validation results");
			final List<String[]> winners = cv.getWinners();
			for (final Future<Void> done : te.experiment(winners)) {done.get();}
		}
		threadExecutor.shutdown();
	}
					
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {new ModelSelection().selection();}
}
