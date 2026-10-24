-- Tables are created as the loader role so it owns them.
SET ROLE loader;

-- restaurant.csv: id,name,city,rating,rating_count,cost,cuisine,lic_no,link,address,menu
CREATE TABLE IF NOT EXISTS raw.restaurants (
  _idx          TEXT,   -- leading index column in the CSV (ignored downstream)
  id            TEXT, name         TEXT, city    TEXT, rating   TEXT,
  rating_count  TEXT, cost         TEXT, cuisine TEXT, lic_no   TEXT,
  link          TEXT, address      TEXT, menu    TEXT
);

-- users.csv: user_id,name,email,password,Age,Gender,Marital Status,
--            Occupation,Monthly Income,Educational Qualifications,Family size
CREATE TABLE IF NOT EXISTS raw.users (
  _idx TEXT,            -- leading index column in the CSV
  user_id TEXT, name TEXT, email TEXT, password TEXT, age TEXT,
  gender TEXT, marital_status TEXT, occupation TEXT, monthly_income TEXT,
  education TEXT, family_size TEXT
);

-- food.csv: f_id,item,veg_or_non_veg
CREATE TABLE IF NOT EXISTS raw.food (
  _idx TEXT,            -- leading index column in the CSV
  f_id TEXT, item TEXT, veg_or_non_veg TEXT
);

-- menu.csv: ,menu_id,r_id,f_id,cuisine,price
CREATE TABLE IF NOT EXISTS raw.menu (
  _idx TEXT,            -- leading index column in the CSV
  menu_id TEXT, r_id TEXT, f_id TEXT, cuisine TEXT, price TEXT
);

-- generated/orders.csv (clean, typed)
CREATE TABLE IF NOT EXISTS raw.orders (
  order_id          BIGINT,
  order_timestamp   TIMESTAMP,
  order_date        DATE,
  user_id           BIGINT,
  r_id              BIGINT,
  restaurant_city   TEXT,
  cuisine           TEXT,
  items_count       INTEGER,
  sales_qty         INTEGER,
  subtotal          NUMERIC(12,2),
  discount          NUMERIC(12,2),
  delivery_fee      NUMERIC(12,2),
  gst               NUMERIC(12,2),
  sales_amount      NUMERIC(12,2),
  currency          TEXT,
  payment_method    TEXT,
  order_status      TEXT,
  customer_rating   NUMERIC,
  delivery_time_min NUMERIC
);

-- generated/order_items.csv (clean, typed)
CREATE TABLE IF NOT EXISTS raw.order_items (
  order_item_id BIGINT,
  order_id      BIGINT,
  r_id          BIGINT,
  f_id          TEXT,
  price         NUMERIC(12,2),
  quantity      INTEGER,
  line_amount   NUMERIC(12,2)
);

-- generated/reviews.csv (clean, typed) - free text for the AI layer
CREATE TABLE IF NOT EXISTS raw.reviews (
  review_id     BIGINT,
  order_id      BIGINT,
  user_id       BIGINT,
  restaurant_id BIGINT,
  rating        NUMERIC,
  comment       TEXT,
  review_date   DATE
);

-- Ingestion tracking
CREATE TABLE IF NOT EXISTS raw.ingestion_log (
  id          BIGSERIAL PRIMARY KEY,
  dataset     TEXT NOT NULL,
  source_key  TEXT,
  rows_loaded BIGINT,
  status      TEXT NOT NULL,
  error       TEXT,
  loaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

RESET ROLE;