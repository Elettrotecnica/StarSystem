package utils;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.regex.Pattern;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

public class Utils {
	
	public static void addInstancesToDataset(
			final Instances instances, 
			final Instances dataset) {
		final Enumeration<Instance> en = instances.enumerateInstances();
		while(en.hasMoreElements()) {dataset.add(en.nextElement());}
	}
	
	public static FilenameFilter getFileFilter(final String regexp) {
		final Pattern pattern = Pattern.compile(regexp);
		return new FilenameFilter() {
		    @Override
		    public boolean accept(File dir, String name) {
		    	return pattern.matcher(name).matches();
		    }
		};
	}
		
	public static Instances useFilter(
			final Instances data, 
			final Filter f) throws Exception {
		f.setInputFormat(data);
		return Filter.useFilter(data, f);
	}
	
	public static String join(final Iterable<String> list, final String separator) {
		String output = "";
		for (String token: list) {
			if (output.length() != 0) {
				output+=separator;
			}; output+=token;
		}; return output;
	}
	
	public static String join(final String[] list, final String separator) {
		return join(Arrays.asList(list), separator);
	}
	
	public static File getEmptyDir (final String dirPath) {
		final File dir = new File(dirPath);
		if (dir.exists()) {
			if (dir.isDirectory()) {
				for(File f : dir.listFiles()) {f.delete();}
				return dir;
			} else {dir.delete();}
		}; dir.mkdir();
		return dir;
	}
	
	public static Instances reorderInstances (
			final Instances data, 
			final List<Future<double[][]>> sortedFeatures) throws InterruptedException, ExecutionException {
		final double[][][] rankings = new double[sortedFeatures.size()][][];
    	int i = 0;
    	for (final Future<double[][]> r : sortedFeatures) {
    		rankings[i] = r.get(); i++;
    	}; return reorderInstances(data, rankings);
	}
	
	private static Instances reorderInstances (
			final Instances data, 
			final double[][][] sortedFeatures) {
		final double[][] avgRanking = Utils.calcAvgRanking(sortedFeatures);
		return reorderInstances(data, avgRanking);
	}
	
	// Apply the sorting received as an array of feature key / feature merit.
	private static Instances reorderInstances (
			Instances data, 
			final double[][] sortedFeatures) {
		final Reorder r = new Reorder();
		
		String indexes = "";
		for (final double [] indxs : sortedFeatures) {
			if (!indexes.equals("")) {indexes+= ",";}
			indexes+= (int)(indxs[0] + 1);
			// Add class variable, which would be excluded otherwise
		} ; indexes+= ",last";
		
		try {r.setAttributeIndices(indexes);} 
		  catch (IllegalArgumentException e) {
			e.printStackTrace();
			return null;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
		try {r.setInputFormat(data);} 
		  catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
		try {data = Filter.useFilter(data, r);} 
		  catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
		return data;
	}
	
	public static Instances keepNFeatures (
			Instances data, 
			final int nFeatures) throws Exception {
		final Remove r = new Remove();
		// From first to nFeatures feature included, plus the class variable
		r.setAttributeIndices("first-"+ nFeatures + ",last");
		r.setInvertSelection(true);
		r.setInputFormat(data);
		data = Filter.useFilter(data, r);
		
		assert(data.numAttributes() == nFeatures + 1);
		
		return data;
	}
	
	static File keepNFeatures (
			final File inFile, 
			final int nFeatures, 
			final String outFileName) throws Exception {
		return keepNFeatures(inFile, nFeatures, new File(outFileName));
	}
	
	static File keepNFeatures (
			final File inFile, 
			final int nFeatures, 
			File outFile) throws Exception {
		if (outFile == null) {
			outFile = File.createTempFile("reducedDataset",".txt");
		}
		Instances data = new Instances(new BufferedReader(new FileReader(inFile)));
		data = keepNFeatures(data, nFeatures);
		data.setRelationName(data.relationName() + "_" + nFeatures + "features");
		return saveInstances(data, outFile);
	}
	
	static File keepNFeatures (final File inFile, final int nFeatures) throws Exception {
		return keepNFeatures(inFile, nFeatures, (File)null);
	}
	
	public static Instances removeAttribute(final Instances data, final String name) throws Exception {
		final Attribute attribute = data.attribute(name);
		if (attribute != null) {
			final Remove r = new Remove();
			r.setAttributeIndices("" + (attribute.index() + 1));
			return useFilter(data, r);
		}
		System.out.println("Warning, couldn't remove missing attribute " + name + " in dataset.");
		return data;
	}
	
	public static Instances readFile (final File f) {
		try {
			final Instances instances = new Instances(new BufferedReader(new FileReader(f)));
			instances.setClassIndex(instances.numAttributes() - 1);
			return instances;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public static Instances readFile (final String f) {
		return readFile(new File(f));
	}
			
	public static File saveInstances (
			final Instances data, 
			final File outFile) throws IOException {
		outFile.createNewFile();
		final BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
		writer.write(data.toString()); writer.flush(); writer.close();
		return outFile;
	}

	public static File saveInstances (
			final Instances data, 
			final String filename) throws IOException {
		final File outFile = new File(filename);
		final String relationName = outFile.getName().replace(".arff", "");
		data.setRelationName(relationName);
		return saveInstances(data, outFile);
	}
		
	static File saveReorderInstances(
			Instances data, 
			final List<Future<double[][]>> rankedAttributes, 
			final String outFileName) 
			throws InterruptedException, ExecutionException, IOException {
		data = Utils.reorderInstances(data, rankedAttributes);
    	return Utils.saveInstances(data, outFileName);
	}
	
	// These comparators allow to sort bi-dimensional arrays by index or value
	final static Comparator<double[]> INDEX_ARRAY_COMPARATOR = new Comparator<double[]>() {
        @Override
        public int compare(final double[] el1, final double[] el2) {
            final double indx1 = el1[0];
            final double indx2 = el2[0];
            if (indx1  > indx2) return 1;
            if (indx1 == indx2) return 0;
            return -1;
        }
    };
    
    public static void sortByIndex(final double[][] ranking) {
    	Arrays.sort(ranking, INDEX_ARRAY_COMPARATOR);
    }

    // Value must be compared in descending order
    private final static Comparator<double[]> VALUE_ARRAY_COMPARATOR = new Comparator<double[]>() {
        @Override
        public int compare(final double[] el1, final double[] el2) {
            final double val1 = el1[1]; 
            final double val2 = el2[1];
            if (val1 <  val2) return 1; 
            if (val1 == val2) return 0; 
            return -1;
        }
    };
            
    public static void sortByValue(final double[][] ranking) {
    	Arrays.sort(ranking, VALUE_ARRAY_COMPARATOR);
    }
    
    static void printRanking(final double[][] ranking) {
    	System.out.println("Keys:");
    	for (final double[] attribute : ranking) {
			System.out.print((int)(attribute[0] + 1) + " ");
		}
    	System.out.println("\nValues:");
    	for (final double[] attribute : ranking) {
			System.out.print(attribute[1] + " ");
		}
    	System.out.println();
    }
	
	private static double[][] calcAvgRanking(final double[][][] rankings) {
        final int nRankings = rankings.length;
        assert (nRankings > 0);
        final int nFeatures = rankings[0].length;
        assert (nFeatures > 0);
        assert (rankings[0][0].length == 2);
        
        // Sort ranking by index, so features are in the 
        // order they were originally in the dataset
		for (int i = 0; i < nRankings; i++) {
			sortByIndex(rankings[i]);
		}

		// Sum ranking values for each feature
		final double[][] avgRanking = rankings[0];
		for (int i = 1; i < nRankings; i++) {
			final double[][] ranking = rankings[i];
			for (int j = 0; j < nFeatures; j++) {
				avgRanking[j][1]+= ranking[j][1];
			}
		}
		
		// Average ranking
		for (int j = 0; j < nFeatures; j++) {
			avgRanking[j][1] /= nRankings;
		}
		
		// Sort array by average ranking value
		sortByValue(avgRanking);
			
		return avgRanking;
	}
	
}