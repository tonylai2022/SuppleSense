def get_recommendation(segment):
    mapping = {
        0: "Offer a loyalty program to increase repeat purchases.",
        1: "Send educational content to re-engage low-open-rate customers.",
        2: "Provide a discount bundle tailored to health-conscious buyers."
    }
    return mapping.get(segment, "Learn more about this segment.")