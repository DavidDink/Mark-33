package main;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

public class HistoryManager {
	private List<Entry> entries;
	private float netCost;
	
	public HistoryManager() {
		entries = new ArrayList<>();
	}
	
	public void addEntry(Entry entry) {
		entries.add(entry);
		netCost += entry.getAction().getCost();
	}
	
	public Entry getEntry(int timestamp) {
		for (Entry e : entries) {
			if (e.getTimestamp() == timestamp)
				return e;
		}
		return null;
	}
	
	public boolean removeEntry(int timestamp) {
		// I can do this because only the timestamp matters when
		// comparing entries.
		Entry e = new Entry(timestamp);
		boolean removed = entries.remove(e);
		if (removed)
			netCost -= e.getAction().getCost();
		return removed;
	}
	
	public List<Entry> getEntries() {
		return Collections.unmodifiableList(entries);
	}
	
	public float getNetCost() {
		return netCost;
	}
	
	public void printData() {
		System.out.print("Desired inside temp: " + ComfortManager.IDEAL_TEMP);
		System.out.println("  Net Cost: " + netCost);
		System.out.println("-----------------");
		entries.forEach(System.out::println);
	}
	
	public static class Entry {
		private int timestamp;
		private final Session session;
		private final ActionMap action;
		
		public Entry(int timestamp, Session sess, ActionMap action) {
			this.timestamp = timestamp;
			this.session = sess;
			this.action = action;
		}
		
		// Constructor (hack)
		private Entry(int timestamp) {
			this(timestamp, null, null);
		}
		
		@Override
		public boolean equals(Object o) {
			if (o == this)
				return true;
			if (!(o instanceof Entry))
				return false;
			Entry other = (Entry)o;
			return other.timestamp == timestamp;
		}
		
		@Override
		public int hashCode() {
			return timestamp;
		}
		
		@Override
		public String toString() {
			Container container = session.getContainer();
			final float comfortLevel = ComfortManager.evaluateComfortLevel(
					container.insideTemp(), container.insideHumidity());
			return timestamp + "," + ComfortManager.IDEAL_TEMP + "," +
					container.getInsideFeelTemp() + "," + container.insideTemp() + "," +
					container.outsideTemp() + "," + ComfortManager.IDEAL_HUMIDITY +
					"," + container.insideHumidity() + "," + container.outsideHumidity()
					+ "," + action.getState() + "," +
					action.getAction() + "," + comfortLevel + "," + action.getCost();
		}

		public int getTimestamp() {
			return timestamp;
		}
		
		public Session getSession() {
			return session;
		}
		
		public ActionMap getAction() {
			return action;
		}
		
	}
}
