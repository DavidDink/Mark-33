package main;

public class Main {
	public static void main(String[] args) {
		Session sess = new Session();
		sess.run(100);
		Utils.saveToFile(sess);
	}
}
