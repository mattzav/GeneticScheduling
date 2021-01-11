
public class Individual {

	private double fitness;
	private double objF;
	private int[] genotype;
	private double sumA;
	private double sumB;

	public Individual(Individual individual) {
		this.genotype = new int[individual.genotype.length];
		for (int i = 0; i < genotype.length; i++)
			genotype[i] = individual.genotype[i];

		this.fitness = individual.fitness;
		this.objF = individual.objF;
		this.sumA = individual.sumA;
		this.sumB = individual.sumB;
	}

	public Individual(int n) {
		genotype = new int[n];
	}

	public void setJob(int pos, int job) {
		genotype[pos] = job;
	}

	public double getSumA() {
		return sumA;
	}

	public double getSumB() {
		return sumB;
	}

	public int[] getGenotype() {
		return genotype;
	}

	public double getFitness() {
		return fitness;
	}

	public double getObjF() {
		return objF;
	}

	public void setFitness(double fitness) {
		this.fitness = fitness;
	}

	public void setObjF(double objF) {
		this.objF = objF;
	}

	public void setSumA(double sumA) {
		this.sumA = sumA;
	}

	public void setSumB(double sumB) {
		this.sumB = sumB;
	}

	public void swap(int i, int j) {
		int toExchange = genotype[i];
		genotype[i] = genotype[j];
		genotype[j] = toExchange;
	}

	@Override
	public String toString() {
		String toReturn = "";
		for (int i = 0; i < genotype.length; i++) {
			toReturn += genotype[i] + " ";
		}
		toReturn += ", OBJ F = " + objF +  " ";
		toReturn += ", FITNESS = " + fitness + "\n";

		return toReturn;
	}
}