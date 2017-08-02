class App {
    public static void main(String[] args) {
        String dataset = "fleet_cleaned%d.csv";
        try {
            KBSPurchaseRegressor kbsRegressor = new KBSPurchaseRegressor(dataset);
            System.out.println(kbsRegressor.predict());
        } catch (Exception e) {
            System.err.println(e);
        } 
    }
}
