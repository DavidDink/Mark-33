package main;

public class Main {
	public static void main(String[] args) {
		Session sess = new Session();
		Engine lester = new Lester();
		sess.run(lester, 10);
		Utils.saveToFile(sess);
	}
}
