package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

public final class Utils {
	public static final long STANDARD_SEED = 13453463456L;
	public static final Random RANDOM = new Random(STANDARD_SEED);

	private Utils() {}
	
	public static boolean withinRange(float val, float desiredVal, float acceptableOffset) {
		return Math.abs(desiredVal - val) <= acceptableOffset;
	}
	
	/**
	 * Returns a random float between -1 and 1
	 */
	public static float standardRandomFloat() {
		return (2f * RANDOM.nextFloat() - 1f);
	}
	
	public static float chooseRandom(float...ops) {
		return ops[RANDOM.nextInt(ops.length)];
	}
	
	public static void saveToFile(Session session) {
		String path = "/Users/David/Desktop/";
		String name = "data" + System.currentTimeMillis() + ".csv";
		File file = new File(path + name);
		StringBuilder builder = new StringBuilder();
		// Labels
		builder.append("time,desired_temp,feel_temp,inside_temp,outside_temp,desired_humidity,"
				+ "inside_humidity,outside_humidity,temp_state,temp_action,humidity_state,"
				+ "humidity_action,comfort_penalty,cost,net_cost,penalty\n");
		
		// Data
		float cost = 0f;
		for (Interval e : session.getHistoryManager().getIntervals()) {
			builder.append(e);
			// Append net-cost and penalty of interval
			builder.append("," + (cost += e.getCost()) + "," + e.getTotalPenalty());
			builder.append("\n");
		}
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(file);
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		writer.write(builder.toString());
		writer.close();
	}

}
