export interface OfflineHospital {
  id: string;
  name: string;
  address: string;
  phone: string;
  category: "Government Hospital" | "Private Hospital";
  latitude: number;
  longitude: number;
}

// Contact details transcribed from the supplied Chennai empanelled-HCO lists
// (3 January and 7 April 2025). Coordinates are stored locally for offline distance sorting.
export const CHENNAI_HOSPITALS: OfflineHospital[] = [
  { id: "apollo-greams", name: "Apollo Hospital", address: "No. 21, Greams Lane, Chennai - 600 006", phone: "044-28290200", category: "Private Hospital", latitude: 13.0632248, longitude: 80.2515817 },
  { id: "billroth-shenoy", name: "Billroth Hospital", address: "No. 43, Lakshmi Talkies Road, Shenoy Nagar, Chennai - 600 030", phone: "7299104050", category: "Private Hospital", latitude: 13.0757435, longitude: 80.2272588 },
  { id: "csi-kalyani", name: "C.S.I. Kalyani Hospital", address: "No. 15, Dr. Radhakrishnan Salai, Mylapore, Chennai - 600 004", phone: "044-28475870", category: "Private Hospital", latitude: 13.0432845, longitude: 80.2724013 },
  { id: "dr-mehtas", name: "Dr. Mehtas Hospital", address: "No. 2, McNichols Road, 3rd Lane, Chetpet, Chennai - 31", phone: "42271001", category: "Private Hospital", latitude: 13.0755, longitude: 80.2422 },
  { id: "fortis-malar", name: "Fortis Malar Hospital", address: "No. 52, 1st Main Road, Gandhi Nagar, Adyar, Chennai - 600 020", phone: "044-42892222", category: "Private Hospital", latitude: 13.0101759, longitude: 80.2586997 },
  { id: "government-general", name: "Government General Hospital", address: "Poonamallee High Road, Park Town, Chennai - 600 003", phone: "044-25305000", category: "Government Hospital", latitude: 13.0806077, longitude: 80.2773314 },
  { id: "government-kilpauk", name: "Government Kilpauk Hospital", address: "Kilpauk, Chennai - 600 010", phone: "044-28364951", category: "Government Hospital", latitude: 13.0782751, longitude: 80.2438802 },
  { id: "thoracic-medicine", name: "Government Hospital for Thoracic Medicine", address: "Tambaram Sanatorium, Chennai - 47", phone: "044-22418450", category: "Government Hospital", latitude: 12.9381279, longitude: 80.1345359 },
  { id: "kasturba-women", name: "Government Kasturba Hospital for Women", address: "Triplicane, Chennai - 600 005", phone: "044-28545449", category: "Government Hospital", latitude: 13.0604285, longitude: 80.2750704 },
  { id: "government-multi-super", name: "Government Multi Super Speciality Hospital", address: "Omandurar Government Estate, Chennai - 600 002", phone: "044-25666000", category: "Government Hospital", latitude: 13.0694592, longitude: 80.273745 },
  { id: "government-royapettah", name: "Government Royapettah Hospital", address: "No. 1, West Cott Road, Chennai - 600 014", phone: "044-28483051", category: "Government Hospital", latitude: 13.0544912, longitude: 80.2651496 },
  { id: "government-stanley", name: "Government Stanley Hospital", address: "Old Jail Road, Chennai - 600 001", phone: "044-25281347", category: "Government Hospital", latitude: 13.1070229, longitude: 80.2852656 },
  { id: "child-health", name: "Institute of Child Health and Hospital for Children", address: "Halls Road, Egmore, Chennai - 600 008", phone: "044-28191135", category: "Government Hospital", latitude: 13.0735496, longitude: 80.257025 },
  { id: "madras-medical-mission", name: "Madras Medical Mission Hospital", address: "4-A, Dr. J. Jayalalitha Nagar, Mogappair, Chennai - 600 037", phone: "26565961", category: "Private Hospital", latitude: 13.0859961, longitude: 80.1870589 },
  { id: "miot", name: "MIOT Hospital", address: "4/112, Mount Poonamallee Road, Manapakkam, Chennai - 600 089", phone: "22492288", category: "Private Hospital", latitude: 13.0208127, longitude: 80.1838081 },
  { id: "rsrm", name: "R.S.R.M. Lying-In Hospital", address: "Cemetery Road, Chennai - 600 013", phone: "044-25902961", category: "Government Hospital", latitude: 13.108941, longitude: 80.2887366 },
  { id: "sankara-nethralaya", name: "Sankara Nethralaya", address: "No. 18, College Road, Nungambakkam, Chennai - 600 006", phone: "28271616", category: "Private Hospital", latitude: 13.0670837, longitude: 80.2505444 },
  { id: "sooriya", name: "Sooriya Hospital", address: "Kodambakkam Sriperumbudur Road, Vadapalani, Chennai - 26", phone: "23761750", category: "Private Hospital", latitude: 13.0481888, longitude: 80.2035199 },
  { id: "sri-ramachandra", name: "Sri Ramachandra Medical College and Research Centre", address: "Porur, Chennai", phone: "24768402", category: "Private Hospital", latitude: 13.0378713, longitude: 80.1402023 },
  { id: "st-isabella", name: "St. Isabella Hospital", address: "Oliver Road, Mylapore, Chennai - 600 004", phone: "24991081", category: "Private Hospital", latitude: 13.0384829, longitude: 80.2621353 },
  { id: "sundaram", name: "Sundaram Medical Foundation", address: "9C, Santhi Colony, 4th Avenue, Anna Nagar, Chennai - 600 040", phone: "26268844", category: "Private Hospital", latitude: 13.08217, longitude: 80.2069819 },
  { id: "vijaya", name: "Vijaya Hospital", address: "175 and 180 NSK Salai, Vadapalani, Chennai - 600 026", phone: "24801642", category: "Private Hospital", latitude: 13.054729, longitude: 80.2110462 },
  { id: "csi-rainy", name: "C.S.I. Rainy Multispeciality Hospital", address: "45, G.A. Road, Old Washermenpet, Chennai - 600 021", phone: "044-40405509", category: "Private Hospital", latitude: 13.1133, longitude: 80.2876 },
  { id: "chennai-nobel", name: "Chennai Nobel Hospitals", address: "4, Audiappa Street, Purasawalkam, Chennai - 600 084", phone: "044-26402222", category: "Private Hospital", latitude: 13.086, longitude: 80.2545 },
  { id: "soundarapandian", name: "Soundarapandian Bone and Joint Hospital", address: "AA-16, 3rd Main Road, Anna Nagar, Chennai - 600 040", phone: "044-43007363", category: "Private Hospital", latitude: 13.0838202, longitude: 80.2156412 },
  { id: "frontier-lifeline", name: "Frontier Lifeline Hospital", address: "R-30-D, Ambattur Industrial Road, Mogappair, Chennai - 600 101", phone: "044-42017575", category: "Private Hospital", latitude: 13.0875243, longitude: 80.1857322 },
  { id: "parvathy-ortho", name: "Parvathy Ortho Hospital", address: "241, GST Road, Chrompet, Chennai - 600 044", phone: "044-22384550", category: "Private Hospital", latitude: 12.9516, longitude: 80.14 },
  { id: "kauvery-hcg", name: "Kauvery HCG Cancer Centre", address: "MBC Towers, 199/90 Luz Church Road, Mylapore, Chennai - 600 004", phone: "9280876715", category: "Private Hospital", latitude: 13.0353, longitude: 80.2565 },
  { id: "kkr-ent", name: "KKR ENT Hospital and Research Institute", address: "827, Poonamallee High Road, Kilpauk, Chennai - 600 010", phone: "044-26411444", category: "Private Hospital", latitude: 13.0781955, longitude: 80.2452886 },
  { id: "trinity-acute-care", name: "Trinity Acute Care Hospital", address: "33, Desikan Road, Mylapore, Chennai - 600 004", phone: "044-24671166", category: "Private Hospital", latitude: 13.038, longitude: 80.257 },
  { id: "mahalakshmi-multispecialty", name: "Mahalakshmi Multispecialty Hospital", address: "1/164, Mount Poonamallee High Road, Iyyappanthangal, Chennai - 600 056", phone: "044-48678283", category: "Private Hospital", latitude: 13.038, longitude: 80.135 },
  { id: "dr-rai", name: "Dr. Rai Memorial Medical Centre", address: "526, Century Plaza, Anna Salai, Teynampet, Chennai - 600 018", phone: "9841731370", category: "Private Hospital", latitude: 13.044837, longitude: 80.2475507 },
  { id: "panimalar", name: "Panimalar Medical College Hospital and Research Centre", address: "Varadharajapuram, Poonamallee, Chennai - 600 123", phone: "8610850481", category: "Private Hospital", latitude: 13.0567, longitude: 80.0747 },
  { id: "tagore-medical", name: "Tagore Medical College and Hospital", address: "Rathinamangalam, Melakottaiyur Post, Chennai - 600 127", phone: "044-22299500", category: "Private Hospital", latitude: 12.8603225, longitude: 80.1362282 },
  { id: "medway-kodambakkam", name: "Medway Hospitals", address: "9, 1st Main Road, United India Colony, Kodambakkam, Chennai - 600 024", phone: "044-24734455", category: "Private Hospital", latitude: 13.0535971, longitude: 80.2275376 },
  { id: "mehta-velappanchavadi", name: "Mehta Multispeciality Hospital", address: "50, Poonamallee High Road, Velappanchavadi, Chennai - 600 077", phone: "044-40071405", category: "Private Hospital", latitude: 13.0697, longitude: 80.1476 },
  { id: "kalaignar-centenary", name: "Kalaignar Centenary Super Speciality Hospital", address: "Guindy, Chennai - 600 032", phone: "044-22500108", category: "Government Hospital", latitude: 13.0086685, longitude: 80.2126063 },
  { id: "kamakshi-memorial", name: "Dr. Kamakshi Memorial Hospital", address: "1, Radial Road, Pallikaranai, Chennai - 600 100", phone: "044-66300301", category: "Private Hospital", latitude: 12.9492371, longitude: 80.2091585 },
  { id: "hindu-mission", name: "Hindu Mission Hospital", address: "103, GST Road, Tambaram West, Chennai - 600 045", phone: "9500153695", category: "Private Hospital", latitude: 12.9238319, longitude: 80.1140106 },
  { id: "srm-global", name: "SRM Global Hospitals", address: "Mahatma Gandhi Road, Potheri, SRM Nagar, Kattankulathur, Tamil Nadu - 603 203", phone: "09644496444", category: "Private Hospital", latitude: 12.8231, longitude: 80.0453 },
];
