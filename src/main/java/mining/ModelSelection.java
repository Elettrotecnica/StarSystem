package mining;

import java.util.concurrent.ExecutorService;

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
		// Pass the same executor to the next phase so they queue executions toghether 
		final TestSetEvaluation te = new TestSetEvaluation(m_conf, threadExecutor);
		te.incrementalExperiment(cv.experiment());
		threadExecutor.shutdown();
	}
					
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(final String[] args) throws Exception {new ModelSelection().selection();}
}
