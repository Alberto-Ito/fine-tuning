# Amazon Reviews 2023 categories

The dataset declares 33 catalog categories plus `Unknown` for records that could not be mapped reliably. Counts below are approximate review/rating counts from the official dataset card.

| Category | Reviews | Short description |
|---|---:|---|
| `All_Beauty` | 701.5K | General beauty, fragrance, makeup, nail, and personal-care products. |
| `Amazon_Fashion` | 2.5M | Cross-category Amazon fashion, apparel, and accessories. |
| `Appliances` | 2.1M | Large and small appliances, parts, and maintenance accessories. |
| `Arts_Crafts_and_Sewing` | 9.0M | Art, craft, sewing, knitting, painting, and handmade-production supplies. |
| `Automotive` | 20.0M | Vehicle parts, tools, accessories, and maintenance consumables. |
| `Baby_Products` | 6.0M | Feeding, transport, hygiene, safety, and nursery products. |
| `Beauty_and_Personal_Care` | 23.9M | Cosmetics, skin care, hair care, grooming, and personal care. |
| `Books` | 29.5M | Printed books and related publications. |
| `CDs_and_Vinyl` | 4.8M | Music on CD, vinyl, and related physical formats. |
| `Cell_Phones_and_Accessories` | 20.8M | Phones, cases, chargers, cables, and mobile accessories. |
| `Clothing_Shoes_and_Jewelry` | 66.0M | Apparel, footwear, jewelry, watches, and accessories. |
| `Digital_Music` | 130.4K | Digitally distributed albums, tracks, and music products. |
| `Electronics` | 43.9M | Consumer electronics, computers, audio, video, and accessories. |
| `Gift_Cards` | 152.4K | Physical and digital gift cards and prepaid equivalents. |
| `Grocery_and_Gourmet_Food` | 14.3M | Food, beverages, ingredients, snacks, and pantry products. |
| `Handmade_Products` | 664.2K | Handmade, personalized, and small-batch products. |
| `Health_and_Household` | 25.6M | Household health, wellness, cleaning, and everyday supplies. |
| `Health_and_Personal_Care` | 494.1K | Products from an older or more specific health taxonomy. |
| `Home_and_Kitchen` | 67.4M | Furniture, kitchenware, decor, organization, and household goods. |
| `Industrial_and_Scientific` | 5.2M | Industrial components, lab supplies, safety equipment, measurement, and MRO products. |
| `Kindle_Store` | 25.6M | E-books and digital publications for Kindle. |
| `Magazine_Subscriptions` | 71.4K | Print and digital magazine subscriptions. |
| `Movies_and_TV` | 6.5M | Films and television content on physical formats. |
| `Musical_Instruments` | 3.0M | Instruments, studio equipment, live-sound gear, and accessories. |
| `Office_Products` | 12.8M | Office supplies, stationery, furniture, and business equipment. |
| `Patio_Lawn_and_Garden` | 16.7M | Outdoor furniture, gardening, landscaping, and yard equipment. |
| `Pet_Supplies` | 16.8M | Pet food, health, hygiene, transport, and accessories. |
| `Software` | 1.0M | Packaged software, licenses, utilities, and digital tools. |
| `Sports_and_Outdoors` | 19.6M | Sports, fitness, camping, and outdoor recreation products. |
| `Subscription_Boxes` | 16.2K | Products delivered periodically through subscriptions. |
| `Tools_and_Home_Improvement` | 26.0M | Hand tools, power tools, hardware, electrical, plumbing, and renovation products. |
| `Toys_and_Games` | 16.3M | Toys, board games, puzzles, collectibles, and family recreation. |
| `Video_Games` | 4.6M | Games, consoles, controllers, and gaming accessories. |
| `Unknown` | 63.8M | Unmapped records; this is not a coherent domain. |

## Suggested selection by use case

### General retail

Good high-volume sources include `Home_and_Kitchen`, `Electronics`, `Clothing_Shoes_and_Jewelry`, `Beauty_and_Personal_Care`, and `Sports_and_Outdoors`. Sampling is required so popular products do not dominate.

### B2B catalog and procurement

Start with `Industrial_and_Scientific`, `Tools_and_Home_Improvement`, `Automotive`, `Office_Products`, and selected `Electronics`. These categories support attribute extraction, compatibility analysis, specification checks, and purchase constraints.

### Small local experiments

Smaller categories reduce storage and iteration time, although some provide limited product diversity.

### Categories requiring extra care

- Health, beauty, baby, and food recommendations may create health or safety risks.
- Industrial, automotive, and tool recommendations may create compatibility or physical-safety risks.
- Books, music, movies, and games may contain copyrighted text or media references.
- `Unknown` should not be treated as one domain.

## Analysis notes

- Catalog categories are not perfect domain labels, and misclassification exists.
- Similar historical taxonomies should not be merged before distribution analysis.
- Review count is not the same as product count or effective diversity.
- Create splits by `parent_asin`, not by individual review, to prevent leakage.

Source: [Amazon Reviews 2023 dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
