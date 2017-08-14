package main;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

public class HistoryManager {
	private List<Interval> intervals;
	private float netCost;
	
	public HistoryManager() {
		intervals = new ArrayList<>();
	}
	
	public void addInterval(Interval Interval) {
		intervals.add(Interval);
		netCost += Interval.getCost();
	}
	
	public Interval getInterval(int timestamp) {
		for (Interval e : intervals) {
			if (e.getTimestamp() == timestamp)
				return e;
		}
		return null;
	}
	
	public boolean removeInterval(int timestamp) {
		// I can do this because only the timestamp matters when
		// comparing intervals.
		Interval e = new Interval(timestamp);
		boolean removed = intervals.remove(e);
		if (removed)
			netCost -= e.getCost();
		return removed;
	}
	
	public List<Interval> getIntervals() {
		return Collections.unmodifiableList(intervals);
	}
	
	public float getNetCost() {
		return netCost;
	}
	
	public void printData() {
		System.out.print("Desired inside temp: " + ComfortManager.IDEAL_TEMP);
		System.out.println("  Net Cost: " + netCost);
		System.out.println("-----------------");
		intervals.forEach(System.out::println);
	}
}
